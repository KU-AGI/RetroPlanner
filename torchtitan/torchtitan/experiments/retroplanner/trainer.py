# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer that carries per-token role/span streams into the loss and out to
the metric loggers.

Two things the stock trainer does not do:

1. ``role_ids`` and ``span_ids`` come off the dataloader as per-token streams.
   They are not model inputs -- the stock ``post_dataloading_process`` would
   forward every non-``input`` key to ``Decoder.forward`` -- so they are pulled
   out here and handed to the loss as ``loss_inputs``, which
   ``ChunkedLossWrapper`` chunks along the sequence axis for free.

2. ``forward_backward_step`` discards the loss's metrics. Here the role
   accumulator is drained at logging time, reduced across data-parallel ranks
   and merged into the metrics the trainer already logs, so wandb gets
   ``loss/reasoning``, ``loss/decision``, ``acc/*`` and ``action/exact_match``
   next to the usual loss curve.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import spmd_types as spmd
import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed import distributed_c10d as c10d

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.validate import Validator
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer

from .board_metrics import ActionCapture, metrics_from_counts
from .loss import RoleCrossEntropyLoss, RoleMetricAccumulator
from .state_dict_adapter import find_expert_param, verify_experts_loaded


class RetroValidator(Validator):
    """Validator for batches that carry the extra role/span streams.

    The base implementation forwards every non-``input`` key of the batch to
    the model, which would hand ``role_ids``/``span_ids`` to
    ``Decoder.forward``. They are dropped here, so validation reports the
    plain held-out loss. The per-role split stays a training-time metric:
    the base ``validate`` calls ``loss_fn`` without loss inputs, and reaching
    into that would mean forking the whole method.
    """

    val_metrics: dict[str, float] = {}
    """Metrics from the last validation, merged by the trainer's next log."""

    @dataclass(kw_only=True, slots=True)
    class Config(Validator.Config):
        probe_dataset_path: str | None = None
        """A TRAIN subset to score with the same machinery, logged as
        `trainprobe/*`.  Read against `val/*`: the two are depth-stratified to the
        same mix, so a gap between them is memorisation and not a difference in
        how deep their episodes happen to be."""

        # Declared, not inherited: Configurable binds the owning class to the
        # Config where it is defined, so an inherited Config would build a
        # plain Validator and hand role_ids straight to Decoder.forward.
        pass

    def post_dataloading_process(self, input_dict, labels, model_parts):
        # Kept, not discarded: the base `validate` calls `loss_fn(pred, labels)`
        # with no loss inputs, so without stashing these the held-out pass reports
        # a loss and nothing else -- no per-role split, no action metrics, and
        # therefore no held-out version of the numbers the run is judged on.
        self._streams = (input_dict.pop("role_ids", None),
                         input_dict.pop("span_ids", None))
        return super().post_dataloading_process(input_dict, labels, model_parts)

    def validate(self, model_parts, step: int) -> None:
        """The held-out pass, then optionally the same over a train subset."""
        self.val_metrics = {}
        self._scored_pass(model_parts, step, prefix="val")
        probe = getattr(self.config, "probe_dataset_path", None)
        if probe:
            self._scored_pass(model_parts, step, prefix="trainprobe",
                              dataset_path=probe, quiet=True)

    def _scored_pass(self, model_parts, step: int, *, prefix: str,
                     dataset_path: str | None = None,
                     quiet: bool = False) -> None:
        """Run `validate` once and keep the role and action metrics from it.

        The streams are injected through a shim rather than by forking `validate`,
        and the collectors are swapped for fresh ones so one pass cannot leak into
        another or into the training interval's totals.  Anything unexpected leaves
        the metrics unreported rather than failing the run: a probe is not worth a
        crash.
        """
        inner = _unwrap_loss(self.loss_fn)
        acc = getattr(inner, "accumulator", None)
        cap = getattr(inner, "capture", None)
        if acc is None:
            if prefix == "val":
                super().validate(model_parts, step)
            return

        saved_call, saved_acc, saved_cap = self.loss_fn, acc, cap
        saved_dl, saved_logval = self.dl_config, self.metrics_processor.log_validation
        try:
            inner.accumulator = RoleMetricAccumulator(
                max_spans=acc.max_spans, num_slots=acc.num_slots)
            if cap is not None:
                inner.capture = ActionCapture()
                inner.capture.arm()
            if dataset_path is not None:
                self.dl_config = replace(self.dl_config, dataset_path=dataset_path)
            loss_holder: dict[str, float] = {}
            if quiet:
                # The base pass logs "validation loss" for the step; a second call
                # would overwrite it with the probe's, so the probe's loss is
                # captured under its own name instead.
                def hold(loss: float, step: int, **kw):
                    loss_holder["loss"] = float(loss)
                self.metrics_processor.log_validation = hold

            def shim(pred, labels, *args, **kwargs):
                roles, spans = getattr(self, "_streams", (None, None))
                if roles is not None and "role_ids" not in kwargs:
                    kwargs["role_ids"] = roles
                    kwargs["span_ids"] = spans
                out = saved_call(pred, labels, *args, **kwargs)
                # One shim call is one microbatch (the chunk loop is inside the
                # wrapper), so the boundary closes here.  Without it the
                # accumulator never folds its per-chunk rows -- no role loss or
                # accuracy at all -- and the capture never advances its microbatch
                # counter, so span ids collide across microbatches.
                inner.accumulator.end_microbatch()
                if inner.capture is not None:
                    inner.capture.end_microbatch()
                return out

            self.loss_fn = shim
            super().validate(model_parts, step)
            metrics = RoleMetricAccumulator.metrics_from_state(
                inner.accumulator.state().cpu())
            if inner.capture is not None:
                metrics.update(metrics_from_counts(
                    inner.capture.counts(self.tokenizer)))
            if "loss" in loss_holder:
                metrics["loss"] = loss_holder["loss"]
            self.val_metrics.update({f"{prefix}/{k}": v for k, v in metrics.items()})
        except Exception as exc:                       # noqa: BLE001
            logger.warning("[retroplanner] %s metrics unavailable: %s", prefix, exc)
        finally:
            self.loss_fn = saved_call
            inner.accumulator = saved_acc
            if cap is not None:
                inner.capture = saved_cap
            self.dl_config = saved_dl
            self.metrics_processor.log_validation = saved_logval


class RetroSFTTrainer(Trainer):
    """Trainer for role-aware retrosynthesis-agent SFT."""

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        log_role_metrics: bool = True
        """Report per-role cross-entropy, accuracy and action exact match."""

    # pyrefly: ignore[bad-override-mutable-attribute]
    config: Config

    def __init__(self, config: Config):
        super().__init__(config)
        self._role_accumulator: RoleMetricAccumulator | None = None
        self._action_capture: ActionCapture | None = None
        if config.log_role_metrics:
            self._role_accumulator = _find_role_accumulator(self.loss_fn)
            if self._role_accumulator is None:
                raise ValueError(
                    "log_role_metrics is set but the configured loss is not a "
                    "RoleCrossEntropyLoss (optionally wrapped in "
                    "ChunkedLossWrapper), so there is nothing to report."
                )
            self._action_capture = getattr(
                _unwrap_loss(self.loss_fn), "capture", None
            )
            self._check_parallelism_is_supported()
            self._wrap_metrics_log()
        self._wrap_checkpoint_load()

    def _check_parallelism_is_supported(self) -> None:
        """Refuse the parallelisms that make per-role attribution impossible.

        Tensor parallelism shards the vocabulary, so the logits reaching the loss
        are a DTensor and an argmax over them is not the model's argmax; pipeline
        parallelism drives the loss through its own schedule, which never passes
        the role streams. Both would report role metrics that quietly mean
        something else.
        """
        parallelism = self.config.parallelism
        if parallelism.tensor_parallel_degree > 1:
            raise ValueError(
                "Role metrics need replicated logits: tensor_parallel_degree="
                f"{parallelism.tensor_parallel_degree} makes them vocab-parallel. "
                "Run with tensor_parallel_degree=1 or set log_role_metrics=False."
            )
        if parallelism.pipeline_parallel_degree > 1:
            raise ValueError(
                "Role metrics are computed in the loss, which pipeline "
                "parallelism drives through its own schedule. Disable pipeline "
                "parallelism or set log_role_metrics=False."
            )

    def _wrap_checkpoint_load(self) -> None:
        """Check the pretrained weights actually landed, right after the load.

        The gpt-oss adapter drops unmapped keys without a word in both
        directions, so a mismatched name means the tensors are simply never
        requested and the model trains on its random init. That is silent all
        the way to the end of the run -- see state_dict_adapter.py.
        """
        config = self.config
        model = self.model_parts[0]
        has_experts = find_expert_param(model) is not None
        applicable = (
            config.checkpoint.enable
            and config.checkpoint.initial_load_in_hf
            and has_experts
        )
        logger.warning(
            "[retroplanner] expert-load check: applicable=%s (enable=%s, in_hf=%s, "
            "has_experts=%s)",
            applicable,
            config.checkpoint.enable,
            config.checkpoint.initial_load_in_hf,
            has_experts,
        )
        if not applicable:
            return

        inner_load = self.checkpointer.load

        def load(*args, **kwargs):
            loaded = inner_load(*args, **kwargs)
            logger.warning("[retroplanner] verifying pretrained experts ...")
            verify_experts_loaded(model, config.hf_assets_path)
            return loaded

        self.checkpointer.load = load  # type: ignore[method-assign]

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]],
        labels: torch.Tensor | list[torch.Tensor],
        global_valid_tokens: float,
    ) -> torch.Tensor:
        if self._role_accumulator is None:
            return super().forward_backward_step(
                input_dict=input_dict,
                labels=labels,
                global_valid_tokens=global_valid_tokens,
            )

        if self.parallel_dims.pp_enabled:
            raise ValueError(
                "Role metrics are computed in the loss, which pipeline "
                "parallelism drives through its own schedule. Disable "
                "pipeline parallelism or set log_role_metrics=False."
            )
        assert isinstance(input_dict, dict)
        assert isinstance(labels, torch.Tensor)

        # Arm the action capture only for a step that will be logged.  It copies
        # the decision tokens to host, so arming it every step would put a sync
        # in the hot path for numbers nobody reads.
        capture = self._action_capture
        if capture is not None:
            capture.arm(self.metrics_processor.should_log(self.step))

        loss_inputs = {
            "role_ids": input_dict.pop("role_ids"),
            "span_ids": input_dict.pop("span_ids"),
        }
        inputs, labels, extra_kwargs = self.post_dataloading_process(input_dict, labels)

        assert len(self.model_parts) == 1
        with self.train_context():
            pred = self.model_parts[0](inputs, **extra_kwargs)
            # pyrefly: ignore[unexpected-keyword]
            loss, _ = self.loss_fn(pred, labels, global_valid_tokens, **loss_inputs)
            del pred
            with spmd.no_typecheck():
                loss.backward()

        # Decision spans are numbered per packed sequence, so their token
        # counts must be resolved into action-level hits before the next
        # microbatch reuses the same slots.
        self._role_accumulator.end_microbatch()
        if capture is not None:
            # Span ids restart with every packed sequence, so the capture has to
            # know where one microbatch ends and the next begins.
            capture.end_microbatch()
        return loss

    # -- metric reporting ---------------------------------------------------
    def _wrap_metrics_log(self) -> None:
        """Merge role metrics into whatever the trainer is about to log.

        ``train_step`` builds ``extra_metrics`` and calls the metrics processor
        inline, so the merge point is the processor's ``log``. Wrapping it here
        keeps the change inside this experiment instead of threading a new
        return value through core ``train_step``.
        """
        inner_log = self.metrics_processor.log

        def log(step: int, *args: Any, extra_metrics: dict | None = None, **kwargs):
            merged = dict(extra_metrics or {})
            merged.update(self._drain_role_metrics())
            merged.update(self._drain_action_metrics())
            # The validator cannot log on its own without draining the training
            # interval mid-flight, so it stashes and the next log picks it up.
            # `self.validator` only exists when the config enabled one, so
            # reaching through it crashes any run with validation off -- at the
            # first log, after the first step has already been paid for.
            val = getattr(getattr(self, "validator", None), "val_metrics", None)
            if val:
                merged.update(val)
                self.validator.val_metrics = {}
                for pre in ("val", "trainprobe"):
                    head = (f"{pre}/loss/decision", f"{pre}/acc/decision",
                            f"{pre}/transition/exact", f"{pre}/transition/tree",
                            f"{pre}/action/exact_match", f"{pre}/action/decoded")
                    shown = " ".join(f"{k.split('/', 1)[1]}={val[k]:.4g}"
                                     for k in head if k in val)
                    if shown:
                        logger.info("[retroplanner] %s: %s", pre, shown)
            inner_log(step, *args, extra_metrics=merged, **kwargs)

        self.metrics_processor.log = log  # type: ignore[method-assign]

    def _drain_role_metrics(self) -> dict[str, float]:
        """Reduce the accumulator across ranks, convert to scalars, reset it."""
        accumulator = self._role_accumulator
        if accumulator is None:
            return {}

        state = accumulator.state()
        if self.parallel_dims.dp_cp_enabled:
            loss_mesh = self.parallel_dims.get_optional_mesh("loss")
            if loss_mesh is not None:
                # dist_utils.dist_sum reduces to a python float and so only
                # takes scalars; these totals are a vector, hence the direct
                # collective. Sums are exact across ranks because every entry
                # is an integer count or a sum of per-token NLLs.
                state = funcol.all_reduce(
                    state.to(self.device),
                    reduceOp=c10d.ReduceOp.SUM.name,
                    group=loss_mesh,
                )
        metrics = RoleMetricAccumulator.metrics_from_state(state.cpu())
        accumulator.reset()

        if metrics:
            sl.log_trace_scalar(
                {k: v for k, v in metrics.items() if k.startswith(("loss/", "acc/"))}
            )
        return metrics

    def _drain_action_metrics(self) -> dict[str, float]:
        """Decode the captured actions and score them field by field.

        Summed across ranks, then divided: the capture is rank-local, so
        averaging per-rank rates would weight a rank that saw two actions the
        same as one that saw twenty.
        """
        capture = self._action_capture
        if capture is None:
            return {}
        counts = capture.counts(self.tokenizer)
        capture.reset()
        capture.arm(False)
        if self.parallel_dims.dp_cp_enabled:
            loss_mesh = self.parallel_dims.get_optional_mesh("loss")
            if loss_mesh is not None:
                counts = funcol.all_reduce(
                    counts.to(self.device),
                    reduceOp=c10d.ReduceOp.SUM.name,
                    group=loss_mesh,
                )
        return metrics_from_counts(counts.cpu())


def _unwrap_loss(loss_fn: Any) -> Any:
    """The RoleCrossEntropyLoss itself, through a ChunkedLossWrapper if present."""
    if hasattr(loss_fn, "capture"):
        return loss_fn
    inner = getattr(loss_fn, "loss_fn", None) or getattr(loss_fn, "inner", None)
    return inner if inner is not None else loss_fn


def _find_role_accumulator(loss_fn: Any) -> RoleMetricAccumulator | None:
    """Locate the role accumulator, through ``ChunkedLossWrapper`` if present."""
    if isinstance(loss_fn, RoleCrossEntropyLoss):
        return loss_fn.accumulator
    if isinstance(loss_fn, ChunkedLossWrapper):
        inner = loss_fn.loss_fn
        if isinstance(inner, RoleCrossEntropyLoss):
            return inner.accumulator
        logger.warning(
            "ChunkedLossWrapper wraps %s, not RoleCrossEntropyLoss; "
            "no role metrics will be reported.",
            type(inner).__name__,
        )
    return None
