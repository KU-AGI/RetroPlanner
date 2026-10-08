# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""gpt-oss weight mapping that does not drop the MoE experts.

``GptOssStateDictAdapter.from_hf_map`` names the expert tensors without their
shape suffixes -- ``...inner_experts.mlp1_weight`` -- but the model's
parameters carry them: ``mlp1_weight_EGD``. Nothing reconciles the two, and
both directions of the conversion skip unmapped keys silently
(``to_hf``: ``if abstract_key not in to_hf_map: continue``), so:

  * loading a pretrained checkpoint leaves every expert at its random init.
    ``CheckpointManager.dcp_load`` runs the target state dict through
    ``to_hf`` first, so the expert tensors are never even requested from the
    safetensors files, and ``load_state_dict`` is then handed a dict without
    them. Nothing raises. On gpt-oss-20b that is 17 of 21 billion parameters,
    and the only symptom is a first-step loss that looks a little high.
  * saving to HF writes those tensors out with an empty dtype and an empty
    shape, producing a checkpoint that is a fraction of its real size and that
    safetensors cannot even parse.

The shapes already line up on both sides (the quantized HF reader dequantizes
mxfp4 to exactly the parameter shape), so the fix is only the names.

``verify_experts_loaded`` is the belt to this braces: it re-reads the
pretrained experts and refuses to train if they did not land.
"""

from __future__ import annotations

import glob
import json
import re

import torch

from torchtitan.models.gpt_oss.state_dict_adapter import GptOssStateDictAdapter
from torchtitan.tools.logging import logger

# tt parameter name (with shape suffix) <- the suffix-less name in the stock map
_EXPERT_SUFFIXES = {
    "mlp1_weight": "mlp1_weight_EGD",
    "mlp1_bias": "mlp1_bias_EG",
    "mlp2_weight": "mlp2_weight_EDF",
    "mlp2_bias": "mlp2_bias_ED",
}


class RetroGptOssStateDictAdapter(GptOssStateDictAdapter):
    """``GptOssStateDictAdapter`` with the expert names corrected."""

    def __init__(self, model_config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)

        renamed: dict[str, str] = {}
        for hf_key, tt_key in self.from_hf_map.items():
            tail = tt_key.rsplit(".", 1)[-1] if tt_key else None
            if tail in _EXPERT_SUFFIXES:
                tt_key = tt_key[: -len(tail)] + _EXPERT_SUFFIXES[tail]
            renamed[hf_key] = tt_key
        self.from_hf_map = renamed

        mapped = sum(
            1
            for v in renamed.values()
            if v and v.endswith(tuple(_EXPERT_SUFFIXES.values()))
        )
        if mapped != len(_EXPERT_SUFFIXES):
            raise ValueError(
                "expected to rename exactly "
                f"{len(_EXPERT_SUFFIXES)} expert entries, renamed {mapped}. "
                "The stock adapter's from_hf_map has changed; re-check it."
            )


def _dequantize_mxfp4(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Undo the mxfp4 packing of a single expert slice.

    Each uint8 holds two e2m1 nibbles (low first); ``scales`` is one e8m0
    exponent byte per 32-value group.
    """
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
    b = blocks.to(torch.int16)
    low, high = b & 0xF, (b >> 4) & 0xF

    def nibble(n: torch.Tensor) -> torch.Tensor:
        magnitude = lut[(n & 7).long()]
        return torch.where(n >= 8, -magnitude, magnitude)

    values = torch.stack([nibble(low), nibble(high)], dim=-1)
    values = values.reshape(b.shape[0], b.shape[1], 32)
    exponent = torch.pow(2.0, scales.float() - 127.0).unsqueeze(-1)
    return (values * exponent).reshape(b.shape[0], -1)


def find_expert_param(model: torch.nn.Module, *, layer: int = 0):
    """The layer's gate/up expert weight, whatever wrappers renamed it.

    Activation checkpointing inserts ``_checkpoint_wrapped_module`` into the
    parameter path, so an exact-name lookup silently finds nothing and the
    check would pass a model it never inspected.
    """
    suffix = "moe.routed_experts.inner_experts.mlp1_weight_EGD"
    prefix = f"layers.{layer}."
    for name, param in model.named_parameters():
        if name.startswith(prefix) and name.endswith(suffix):
            return param
    return None


def verify_experts_loaded(
    model: torch.nn.Module,
    hf_assets_path: str,
    *,
    layer: int = 0,
    expert: int = 0,
    min_cosine: float = 0.99,
) -> float:
    """Raise unless the model's experts match the pretrained checkpoint.

    Compares one expert slice against the released mxfp4 weights. A silently
    unmapped expert scores ~0 here, which is the failure this exists to catch;
    a correctly loaded one scores ~1 even after the dequantization round trip.

    Returns the cosine similarity so callers can log it.
    """
    param = find_expert_param(model, layer=layer)
    if param is None:
        raise ValueError(f"no layer-{layer} expert parameter on this model")

    # Under FSDP the parameter is a DTensor holding this rank's shard; the
    # comparison needs the whole tensor.
    from torch.distributed.tensor import DTensor

    if isinstance(param, DTensor):
        param = param.full_tensor()

    snapshots = glob.glob(f"{hf_assets_path.rstrip('/')}/model.safetensors.index.json")
    if not snapshots:
        raise FileNotFoundError(
            f"no model.safetensors.index.json under {hf_assets_path}"
        )
    index = json.load(open(snapshots[0]))["weight_map"]
    root = re.sub(r"model\.safetensors\.index\.json$", "", snapshots[0])

    from safetensors import safe_open

    blocks_key = f"model.layers.{layer}.mlp.experts.gate_up_proj_blocks"
    scales_key = f"model.layers.{layer}.mlp.experts.gate_up_proj_scales"
    with safe_open(root + index[blocks_key], framework="pt") as f:
        blocks = f.get_slice(blocks_key)[expert]
    with safe_open(root + index[scales_key], framework="pt") as f:
        scales = f.get_slice(scales_key)[expert]

    reference = _dequantize_mxfp4(blocks, scales)
    actual = param[expert].detach().to("cpu", torch.float32)
    if actual.shape != reference.shape:
        raise ValueError(
            f"expert shape {tuple(actual.shape)} != pretrained "
            f"{tuple(reference.shape)}; the mapping is wrong, not just the load"
        )

    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual.flatten(), reference.flatten(), dim=0
        )
    )
    if cosine < min_cosine:
        raise ValueError(
            f"MoE experts did not load: cosine to the pretrained weights is "
            f"{cosine:.6f}, below {min_cosine}. Training now would fine-tune "
            "randomly initialised experts -- 17 of 21 billion parameters on "
            "gpt-oss-20b -- and nothing downstream would say so."
        )
    logger.info("Pretrained experts verified: cosine %.6f", cosine)
    return cosine
