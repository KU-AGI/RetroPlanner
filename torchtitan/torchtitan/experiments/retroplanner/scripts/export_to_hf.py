# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export a trained gpt-oss DCP checkpoint as a servable bf16 HF checkpoint.

``scripts/checkpoint_conversion/convert_to_hf.py`` cannot do this for gpt-oss:
its adapter drops every expert tensor (see state_dict_adapter.py), so the
result holds the attention weights with the experts written out as empty
tensors, and safetensors cannot even parse it.

This writes what a server actually wants: experts under their *unquantized*
names, ``model.layers.N.mlp.experts.gate_up_proj``, in the HF orientation
[E, hidden, 2 * intermediate]. torchtitan holds the transpose of that
([E, 2 * intermediate, hidden], matching the mxfp4 block layout the release
ships), so the expert weights are transposed on the way out. config.json is
copied with ``quantization_config`` stripped, because these weights are bf16.

    python -m torchtitan.experiments.retroplanner.scripts.export_to_hf \\
        --checkpoint outputs/checkpoint/step-300 \\
        --assets /path/to/released/snapshot \\
        --dest "$RP_CACHE"/retro-sft-20b-hf
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file

from torchtitan.experiments.retroplanner.state_dict_adapter import (
    RetroGptOssStateDictAdapter,
)
from torchtitan.models.gpt_oss import model_registry

# tt parameter -> (hf name, whether the last two axes need swapping)
_EXPERTS = {
    "mlp1_weight_EGD": ("gate_up_proj", True),
    "mlp1_bias_EG": ("gate_up_proj_bias", False),
    "mlp2_weight_EDF": ("down_proj", True),
    "mlp2_bias_ED": ("down_proj_bias", False),
}
_SHARD_BYTES = 4 * 2**30


def to_hf_name(tt_key: str, inverse: dict[str, str]) -> tuple[str, bool] | None:
    """HF name for a tt parameter, and whether to transpose it."""
    for suffix, (hf_leaf, transpose) in _EXPERTS.items():
        if tt_key.endswith(suffix):
            layer = re.search(r"layers\.(\d+)\.", tt_key)
            if layer is None:
                return None
            return f"model.layers.{layer.group(1)}.mlp.experts.{hf_leaf}", transpose

    abstract = re.sub(r"(\d+)", "{}", tt_key, count=1)
    if abstract in inverse:
        layer_match = re.search(r"\d+", tt_key)
        name = inverse[abstract]
        return (name.format(layer_match.group(0)) if layer_match else name), False
    if tt_key in inverse:
        return inverse[tt_key], False
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path, help="released snapshot")
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--flavor", default="20b")
    args = parser.parse_args()

    spec = model_registry(args.flavor)
    with torch.device("cpu"):
        model = spec.model.build()
    state_dict = model.state_dict()

    print(f"loading {args.checkpoint} ...", flush=True)
    dcp.load(state_dict, checkpoint_id=str(args.checkpoint))

    adapter = RetroGptOssStateDictAdapter(spec.model, str(args.assets))
    inverse = {v: k for k, v in adapter.from_hf_map.items() if v}

    args.dest.mkdir(parents=True, exist_ok=True)
    shard: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    weight_map: dict[str, str] = {}
    total = 0
    shards: list[dict] = []
    skipped: list[str] = []

    def flush() -> None:
        nonlocal shard, shard_bytes
        if not shard:
            return
        shards.append(shard)
        shard = {}
        shard_bytes = 0

    for tt_key in sorted(state_dict):
        mapped = to_hf_name(tt_key, inverse)
        if mapped is None:
            skipped.append(tt_key)
            continue
        hf_key, transpose = mapped
        value = state_dict[tt_key].detach()
        if transpose:
            value = value.transpose(1, 2).contiguous()
        value = value.to(torch.bfloat16)
        shard[hf_key] = value
        shard_bytes += value.numel() * value.element_size()
        total += value.numel() * value.element_size()
        if shard_bytes >= _SHARD_BYTES:
            flush()
    flush()

    for i, tensors in enumerate(shards):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(tensors, str(args.dest / name), metadata={"format": "pt"})
        for key in tensors:
            weight_map[key] = name
        print(f"  {name}: {len(tensors)} tensors", flush=True)

    (args.dest / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1
        )
    )

    config = json.loads((args.assets / "config.json").read_text())
    config.pop("quantization_config", None)
    config["dtype"] = "bfloat16"
    (args.dest / "config.json").write_text(json.dumps(config, indent=2))
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
    ):
        source = args.assets / name
        if source.exists():
            shutil.copy2(source, args.dest / name)

    print(f"wrote {len(weight_map)} tensors, {total / 2**30:.1f} GiB -> {args.dest}")
    # expert_bias_E is torchtitan's own load-balancing buffer and has no HF
    # counterpart; anything else here means a mapping went missing.
    unexpected = [k for k in skipped if not k.endswith("expert_bias_E")]
    print(f"skipped {len(skipped)} tt keys; unexpected: {unexpected or 'none'}")


if __name__ == "__main__":
    main()
