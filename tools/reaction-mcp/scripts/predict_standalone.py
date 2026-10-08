import argparse
import json
import logging
import sys
from typing import List, Dict, Any
import os
import asyncio
import threading
import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn


# Limit multiprocessing for problematic models
# setdefault, not assignment: a pool of N replicas on one machine wants a FEW
# threads each, so the launcher's value must win. Oversubscribing cores shows up
# as CPU load rather than throughput. The model still runs on the GPU; these
# threads are the tokenise/RDKit side.
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --- torch>=2.6 compatibility shim -------------------------------------------------
# syntheseus model checkpoints (LocalRetro, RootAligned) were saved
# with pickled custom classes and load fine only with weights_only=False, which was
# the default before torch 2.6. Newer torch (2.6+, required for Blackwell GPUs)
# flipped the default to True, breaking these loads. Restore the old default.
try:
    import torch as _torch

    _orig_torch_load = _torch.load

    def _torch_load_compat(*args, **kwargs):
        # Force weights_only=False: a checkpoint loader that passes weights_only=True
        # explicitly would not be overridden by setdefault. These are trusted
        # syntheseus checkpoints.
        kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)

    _torch.load = _torch_load_compat
except Exception:  # torch not importable yet in some envs; models import it lazily
    pass
# -----------------------------------------------------------------------------------


# Global variable to store model name for logging
MODEL_NAME = None

# Models whose prediction path draws from the global `random` module at inference time
# (test-time SMILES augmentation). These need per-molecule seeding to be reproducible;
# see the RETRO_DETERMINISTIC_PREDICT block in get_syntheseus_model().
AUGMENTED_MODELS = {"root_aligned"}

# Global profiling state
PROFILING_ENABLED = False
PROFILER_INSTANCE = None

def setup_logging(model_name: str):
    """Configure logging with model name prefix"""
    global MODEL_NAME
    MODEL_NAME = model_name
    
    # Create custom formatter that includes model name
    formatter = logging.Formatter(
        f'%(asctime)s - [{model_name.upper()}] - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler with custom formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Configure uvicorn loggers with our custom formatter
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(logging.INFO)
    uvicorn_logger.handlers = []
    uvicorn_logger.addHandler(console_handler)
    
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.setLevel(logging.INFO)
    uvicorn_access_logger.handlers = []
    uvicorn_access_logger.addHandler(console_handler)
    
    # Prevent uvicorn loggers from propagating to root (avoid duplicate logs)
    uvicorn_logger.propagate = False
    uvicorn_access_logger.propagate = False
    
    return logging.getLogger(__name__)

def get_uvicorn_log_config(model_name: str):
    """Create uvicorn logging configuration with model name"""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": f"%(asctime)s - [{model_name.upper()}] - %(name)s - %(levelname)s - %(message)s",
            },
            "access": {
                "format": f"%(asctime)s - [{model_name.upper()}] - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
        },
    }

def get_syntheseus_model(model_name: str):
    """Loads a pretrained Syntheseus model by name."""
    logger = logging.getLogger(__name__)
    try:
        # Only the two single-step models RetroPlanner uses: root_aligned (R-SMILES)
        # and localretro.
        from syntheseus.reaction_prediction.inference import (
            LocalRetroModel,
            RootAlignedModel,
        )
        models = {
            "localretro": LocalRetroModel,
            "root_aligned": RootAlignedModel,
        }
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not recognized.")
        
        # RootAligned canonicalises its beam output with
        # `multiprocessing.Pool(multiprocessing.cpu_count())` (root_aligned.py:225) --
        # a pool FORKED AND TORN DOWN ON EVERY PREDICTION, sized to the whole machine.
        # That is one fork of a large process per core per call, and a fleet of replicas
        # turns it into a fork storm in which almost no time is spent in the model. What
        # the pool parallelises is a few hundred RDKit canonicalisations, cheap to run
        # serially. So it is replaced with an in-process serial map (RETRO_MP_WORKERS>1
        # restores a real pool of that fixed size). Scoped to this model.
        if model_name == "root_aligned":
            import multiprocessing as _mp
            _n = int(os.getenv("RETRO_MP_WORKERS", "1"))
            if _n > 1:
                _mp.cpu_count = lambda: _n
            else:
                class _SerialPool:
                    def __init__(self, *a, **k): pass
                    def map(self, func, iterable, chunksize=None): return [func(x) for x in iterable]
                    def imap(self, func, iterable, chunksize=None): return (func(x) for x in iterable)
                    def close(self): pass
                    def join(self): pass
                    def terminate(self): pass
                    def __enter__(self): return self
                    def __exit__(self, *a): return False
                _mp.Pool = _SerialPool
                _mp.cpu_count = lambda: 1
            logger.info(f"'root_aligned': per-call multiprocessing pool capped at {_n} "
                        f"(serial map if 1) -- see the fork-storm note in this file")

        logger.info(f"Loading Syntheseus model: {model_name}...")
        model_instance = models[model_name](use_cache=False, default_num_results=10)
        # syntheseus's inference wrappers never call .eval(). LocalRetro then keeps its
        # Dropout layers live at prediction time, so the same molecule comes back with a
        # different top-k ordering on every call -- which silently voids the deterministic
        # protocol (RETRO_ML_SHUFFLE=0 / PYTHONHASHSEED=0) the searches rely on.
        #
        # The sweep RECURSES: RootAligned keeps its network at `m.translator.model`, two
        # levels down, so a flat pass over vars() reports "0 modules" and silently checks
        # nothing. Depth 4 covers every wrapper here.
        try:
            import torch as _t
            _found = []

            def _sweep(obj, path="model", depth=0, seen=None):
                seen = set() if seen is None else seen
                if id(obj) in seen or depth > 4:
                    return
                seen.add(id(obj))
                if isinstance(obj, _t.nn.Module):
                    obj.eval()
                    _found.append(path)
                    return
                if isinstance(obj, (list, tuple)):
                    for i, v in enumerate(obj[:8]):
                        _sweep(v, f"{path}[{i}]", depth + 1, seen)
                elif isinstance(obj, dict):
                    for k, v in list(obj.items())[:32]:
                        _sweep(v, f"{path}[{k!r}]", depth + 1, seen)
                elif hasattr(obj, "__dict__"):
                    for k, v in list(vars(obj).items())[:64]:
                        _sweep(v, f"{path}.{k}", depth + 1, seen)

            _sweep(model_instance)
            logger.info(f"eval() applied to {len(_found)} nn.Module(s) of "
                        f"'{model_name}': {', '.join(_found) or 'none found'}")
        except Exception as _e:
            logger.warning(f"eval() sweep failed for '{model_name}': {_e}")

        # Test-time augmentation makes some models nondeterministic even in eval mode.
        # RootAligned picks `num_augmentations` SMILES root atoms per molecule with
        # `random.sample`/`random.choices` off the GLOBAL, UNSEEDED random module
        # (syntheseus root_aligned.py:177-181), so the same molecule can return a different
        # candidate set and different vote-aggregated scores on every call. Worse, the RNG
        # advances inside the per-input loop, so a molecule's augmentation depends on its
        # POSITION IN THE BATCH.
        #
        # Fix both by seeding from the molecule's own canonical SMILES and handing the model
        # one molecule at a time. The seed is a content hash, not hash(), because hash() is
        # salted per process even under PYTHONHASHSEED=0 for non-str types and would make
        # the value depend on the server, not the molecule. Costs the cross-molecule batch
        # (RootAligned still batches its 20 augmentations internally).
        if os.getenv("RETRO_DETERMINISTIC_PREDICT", "1") != "0" and model_name in AUGMENTED_MODELS:
            import hashlib as _hl
            import random as _rnd
            _orig = type(model_instance)._get_reactions

            def _deterministic(inputs, num_results, *args, **kwargs):
                out = []
                for _mol in inputs:
                    _seed = int.from_bytes(
                        _hl.blake2b(_mol.smiles.encode(), digest_size=8).digest(), "big")
                    _rnd.seed(_seed)
                    out.extend(_orig(model_instance, [_mol], num_results, *args, **kwargs))
                return out

            model_instance._get_reactions = _deterministic
            logger.info(f"'{model_name}': per-molecule deterministic seeding enabled "
                        f"(augmentation RNG pinned to the canonical SMILES).")
        if os.path.exists(f"./{model_name}-cache.pkl"):
            import pickle
            model_instance._cache = pickle.load(open(f"./{model_name}-cache.pkl", "rb"))
            logger.info(f"Loaded cache for {model_name}.")
        logger.info(f"Syntheseus model '{model_name}' loaded successfully.")
        return model_instance
    except ImportError:
        logger.error("Syntheseus is not installed. Please install it in the 'syntheseus-full' environment.")
        raise
    except Exception as e:
        logger.error(f"Failed to load Syntheseus model '{model_name}': {e}")
        raise

def get_inventory():
    """Loads the RetroStar inventory for checking starting materials."""
    logger = logging.getLogger(__name__)
    try:
        import pickle
        logger.info("Loading RetroStar inventory...")
        inventory = pickle.load(open("./full_inventory.pkl", "rb"))
        logger.info("RetroStar inventory loaded successfully.")
        return inventory
    except ImportError:
        logger.error("syntheseus_retro_star_benchmark is not installed.")
        raise
    except Exception as e:
        logger.error(f"Failed to load RetroStar inventory: {e}")
        raise

async def check_purchasable(inventory: Any, smiles_list: List[str]) -> Dict[str, bool]:
    """
    Checks if molecules are purchasable using the inventory.
    """
    from syntheseus import Molecule
    logger = logging.getLogger(__name__)
    
    try:
        results = {}
        for smiles in smiles_list:
            try:
                mol = Molecule(smiles=smiles)
                results[smiles] = inventory.is_purchasable(mol)
            except Exception as e:
                logger.warning(f"Error checking purchasability for {smiles}: {e}")
                results[smiles] = False
        return results

    except Exception as e:
        logger.error(f"Error during purchasability check: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



# Assuming syntheseus and torch are installed
# from syntheseus import Molecule
# import torch

# --- Configuration ---
MAX_BATCH_SIZE = 16
BATCH_TIMEOUT = 1.0  # 1 second

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Pydantic Models for API ---
class PredictionInput(BaseModel):
    smiles: str
    top_n: int = 10

class PurchasabilityInput(BaseModel):
    smiles_list: List[str]


# --- Core Batching Components ---

@dataclass
class RequestItem:
    """A dataclass to hold request data, its completion event, and result."""
    smiles: str
    top_n: int
    event: asyncio.Event = field(default_factory=asyncio.Event)
    result: Any = None
    profiler: Any = None  # Optional profiler instance for this request

# Global asyncio queue, will be initialized on startup
request_queue: asyncio.Queue[RequestItem] = None


def run_prediction_batch(model: Any, requests: List[tuple], profiler: Any = None) -> List[List[Dict[str, Any]]]:
    """
    Process a batch of prediction requests synchronously.
    """
    # NOTE: This function is synchronous and will be run in a thread pool.
    from syntheseus import Molecule  # Keep imports local for threaded execution
    logger.info(f"⚙️  Processing batch of {len(requests)} requests in a worker thread.")
    
    # Start profiling within the thread if profiler is provided
    if profiler is not None:
        try:
            profiler.start()
            logger.info("🔍 Started profiling within worker thread")
        except Exception as e:
            logger.warning(f"Failed to start profiler in thread: {e}")
            profiler = None
    
    try:
        if not requests:
            return []

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        molecules = []
        top_n_values = []
        for smiles, top_n in requests:
            try:
                mol = Molecule(smiles=smiles)
                molecules.append(mol)
                top_n_values.append(top_n)
            except Exception as e:
                logger.warning(f"Invalid SMILES '{smiles}': {e}")
                molecules.append(None)
                top_n_values.append(top_n)

        valid_molecules = [mol for mol in molecules if mol is not None]
        batch_predictions = []
        if valid_molecules:
            try:
                max_top_n = max(t for m, t in zip(molecules, top_n_values) if m is not None)
                import contextlib
                try:
                    import torch as _t
                    _ng = _t.no_grad()
                except ImportError:
                    _ng = contextlib.nullcontext()
                with _ng:
                    batch_predictions = model(valid_molecules, num_results=max_top_n)
            except Exception as e:
                logger.error(f"🔥 Model batch prediction failed: {e}", exc_info=True)
                # Ensure batch_predictions has the correct length for the result mapping
                batch_predictions = [[] for _ in valid_molecules]

        results = []
        valid_idx = 0
        for i, (mol, top_n) in enumerate(zip(molecules, top_n_values)):
            if mol is None:
                results.append([])
                continue

            if valid_idx < len(batch_predictions) and batch_predictions[valid_idx]:
                predictions = batch_predictions[valid_idx]
                individual_results = []
                for ss_result in predictions[:top_n]:
                    try:
                        precursors_mols = ss_result.reactants
                        precursors_smiles = [mol.smiles for mol in precursors_mols]
                        confidence = ss_result.metadata.get("probability", 0.0)
                        individual_results.append({
                            "precursors": precursors_smiles,
                            "confidence": float(confidence),
                        })
                    except Exception as e:
                        logger.warning(f"Error extracting result for '{mol.smiles}': {e}")
                
                individual_results.sort(key=lambda x: x["confidence"], reverse=True)
                results.append(individual_results)
            else:
                results.append([])
            
            valid_idx += 1
        
        logger.info(f"✅ Batch processing complete.")
        return results
    
    finally:
        # Stop profiling within the thread if profiler is provided
        if profiler is not None:
            try:
                profiler.stop()
                logger.info("🔍 Stopped profiling within worker thread")
            except Exception as e:
                logger.warning(f"Failed to stop profiler in thread: {e}")


async def batch_processing_worker(model: Any):
    """
    The core background task that handles dynamic batching.
    It runs in a continuous loop, pulling requests from the queue.
    """
    while True:
        batch: List[RequestItem] = []
        start_time = asyncio.get_event_loop().time()

        # Gather requests until the batch is full or a timeout occurs
        while len(batch) < MAX_BATCH_SIZE:
            try:
                # Calculate remaining time to wait
                remaining_time = BATCH_TIMEOUT - (asyncio.get_event_loop().time() - start_time)
                if remaining_time <= 0:
                    break
                
                # Wait for a new item from the queue
                item = await asyncio.wait_for(request_queue.get(), timeout=remaining_time)
                batch.append(item)
            except asyncio.TimeoutError:
                break # Timeout reached, process the current batch

        if not batch:
            continue

        # Prepare the data for the synchronous batch processing function
        requests_to_process = [(item.smiles, item.top_n) for item in batch]
        current_batch = batch # Keep a reference to the batch for the finally block
        
        # Check if any request in the batch has profiling enabled
        profiler_for_batch = None
        profiled_items = [item for item in batch if hasattr(item, 'profiler') and item.profiler is not None]
        if profiled_items:
            # Use the first profiler found (in practice, there should only be one profiled request per batch)
            profiler_for_batch = profiled_items[0].profiler
        
        try:
            # Run the blocking, synchronous function in a separate thread
            batch_results = await asyncio.to_thread(
                run_prediction_batch, model, requests_to_process, profiler_for_batch
            )

            # Distribute results back to the waiting API requests
            for item, result_data in zip(current_batch, batch_results):
                item.result = result_data
        except Exception as e:
            logger.error(f"An unexpected error occurred in the batch worker: {e}")
            # Propagate the exception to all waiting requests
            for item in current_batch:
                item.result = e
        finally:
            # Wake up all waiting request handlers
            for item in current_batch:
                item.event.set()


def create_app(model: Any, model_name: str, inventory: Any = None) -> FastAPI:
    """Creates a FastAPI app for a given Syntheseus model."""
    app = FastAPI(
        title=f"Syntheseus Model Server ({model_name})",
        version="1.0.0",
    )
    
    # Store model and inventory in app state for access in startup event
    app.state.model = model
    app.state.model_name = model_name
    app.state.inventory = inventory
    app.state.profiling_enabled = False
    app.state.profiler_results = []

    # Profiling runs inside the threaded execution, where the model work happens.

    @app.on_event("startup")
    async def startup_event():
        """On server startup, create the background task for batch processing."""
        global request_queue
        request_queue = asyncio.Queue()
        logger.info("🚀 Server starting up. Initializing batch processing worker...")
        asyncio.create_task(batch_processing_worker(app.state.model))

    @app.post("/predict", response_model=List[Dict[str, Any]])
    async def predict_endpoint(data: PredictionInput, profile: bool = Query(False, description="Enable profiling for this request")):
        """Runs single-step retrosynthesis with dynamic batching."""
        # Create profiler instance if profiling is requested (either via parameter or global setting)
        profiler_instance = None
        should_profile = profile or app.state.profiling_enabled
        
        if should_profile:
            profiler_instance = Profiler()
            
        try:
            # 1. Create a request item with a unique event for this request
            request_item = RequestItem(
                smiles=data.smiles, 
                top_n=data.top_n,
                profiler=profiler_instance
            )

            # 2. Add the item to the global queue
            await request_queue.put(request_item)

            # 3. Wait until the batch worker processes this request and sets the event
            await request_item.event.wait()
            
            # 4. Return the result (or raise an error if one occurred)
            if isinstance(request_item.result, Exception):
                raise HTTPException(status_code=500, detail=f"Prediction failed: {request_item.result}")
            
            return request_item.result
        finally:
            # Handle profiler results after the threaded execution is complete
            if profiler_instance is not None:
                try:
                    profile_output = profiler_instance.output_text(unicode=True, color=True)
                    profile_html = profiler_instance.output_html()
                    
                    # Store the profile result
                    profile_data = {
                        "timestamp": time.time(),
                        "path": "/predict",
                        "method": "POST",
                        "smiles": data.smiles,
                        "top_n": data.top_n,
                        "profile_text": profile_output,
                        "profile_html": profile_html
                    }
                    app.state.profiler_results.append(profile_data)
                    
                    # Keep only the last 10 profile results
                    if len(app.state.profiler_results) > 10:
                        app.state.profiler_results = app.state.profiler_results[-10:]
                        
                    logger.info(f"🔍 Profiling completed for SMILES: {data.smiles}")
                    # Also log the profile output to console for individual requests
                    if profile:
                        logger.info(f"Profile output:\n{profile_output}")
                except Exception as e:
                    logger.error(f"Error processing profiler results: {e}")

    @app.post("/check_purchasable", response_model=Dict[str, bool])
    async def check_purchasable_endpoint(data: PurchasabilityInput):
        """Check if molecules are in the starting material library."""
        if app.state.inventory is None:
            raise HTTPException(status_code=503, detail="Inventory not available")
        
        # Loop over `is_purchasable` in a thread to avoid blocking the event loop.
        def _check():
            results = {}
            from syntheseus import Molecule
            for smiles in data.smiles_list:
                try:
                    mol = Molecule(smiles=smiles)
                    results[smiles] = app.state.inventory.is_purchasable(mol)
                except Exception:
                    results[smiles] = False
            return results

        return await asyncio.to_thread(_check)

    @app.get("/health")
    def health_check():
        return {"status": "ok", "model_name": app.state.model_name}

    @app.get("/save_cache")
    async def save_cache():
        """Save the cache to a file."""
        try:
            import pickle
            model_name = app.state.model_name
            with open(f"./{model_name}-cache.pkl", "wb+") as f:
                pickle.dump(app.state.model._cache, f)
            return {"status": "ok"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving cache: {str(e)}")

    @app.post("/profiling/enable")
    async def enable_profiling():
        """Enable automatic profiling for all predict requests."""
        app.state.profiling_enabled = True
        logger.info("🔍 Automatic profiling enabled for all predict requests")
        return {"status": "enabled", "message": "Profiling enabled for all predict requests"}

    @app.post("/profiling/disable")
    async def disable_profiling():
        """Disable automatic profiling."""
        app.state.profiling_enabled = False
        logger.info("🔍 Automatic profiling disabled")
        return {"status": "disabled", "message": "Profiling disabled"}

    @app.get("/profiling/status")
    async def profiling_status():
        """Get current profiling status."""
        return {
            "enabled": app.state.profiling_enabled,
            "results_count": len(app.state.profiler_results)
        }

    @app.get("/profiling/results")
    async def get_profile_results():
        """Get all stored profile results."""
        return {
            "count": len(app.state.profiler_results),
            "results": [
                {
                    "timestamp": result["timestamp"],
                    "path": result["path"],
                    "method": result["method"],
                    "smiles": result.get("smiles", "N/A"),
                    "top_n": result.get("top_n", "N/A"),
                    "profile_text": result["profile_text"]
                }
                for result in app.state.profiler_results
            ]
        }

    @app.get("/profiling/results/latest/html", response_class=HTMLResponse)
    async def get_latest_profile_html():
        """Get the latest profile result as HTML."""
        if not app.state.profiler_results:
            return HTMLResponse("<html><body><h1>No profile results available</h1></body></html>")
        
        latest_result = app.state.profiler_results[-1]
        return HTMLResponse(latest_result["profile_html"])

    @app.delete("/profiling/results")
    async def clear_profile_results():
        """Clear all stored profile results."""
        count = len(app.state.profiler_results)
        app.state.profiler_results.clear()
        logger.info(f"🗑️ Cleared {count} profile results")
        return {"status": "cleared", "cleared_count": count}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a standalone MCP server for a Syntheseus model."
    )
    parser.add_argument("model_name", type=str, help="root_aligned (R-SMILES), localretro, or inventory.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for the server.")
    parser.add_argument("--port", type=int, default=8002, help="Port for the server.")
    parser.add_argument("--enable-profiling", action="store_true", help="Enable automatic profiling on startup.")
    
    args = parser.parse_args()
    
    # Setup logging with model name
    logger = setup_logging(args.model_name)
    
    try:
        if args.model_name == "inventory":
            inventory = get_inventory()
            app = create_app(inventory, args.model_name, inventory)
            logger.info(f"Starting server for model '{args.model_name}' on http://{args.host}:{args.port}")
        else:
            model = get_syntheseus_model(args.model_name)
            inventory = None
            app = create_app(model, args.model_name, inventory)
            
            logger.info(f"Starting server for model '{args.model_name}' on http://{args.host}:{args.port}")
        
        # Enable profiling if requested
        if args.enable_profiling:
            app.state.profiling_enabled = True
            logger.info("🔍 Automatic profiling enabled via command line flag")
        
        # Use custom logging config and prevent uvicorn from spawning worker processes
        log_config = get_uvicorn_log_config(args.model_name)
        uvicorn.run(app, host=args.host, port=args.port, workers=1, reload=False, log_config=log_config)

    except Exception as e:
        logger.error(f"Failed to start server for model {args.model_name}: {e}")
        sys.exit(1) 