# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Write a bf16 copy of a released gpt-oss checkpoint, experts included.

The released weights store the MoE experts in mxfp4 (blocks of packed 4-bit
values plus an e8m0 scale per 32 values). torchtitan can nominally read that
through ``QuantizedHuggingFaceStorageReader``, but with the torch this runs on
it fills only part of each expert tensor and zeroes the rest, and zeroed
experts silence the MoE entirely.

So the dequantization happens here instead, once, and the result is a plain
bf16 checkpoint that loads through the ordinary reader
(``initial_load_in_hf_quantized=False``).

Tensor names and layout are kept exactly as the source has them -- experts stay
under ``...gate_up_proj_blocks`` in the native [E, G, D] orientation, which is
what torchtitan's parameters expect. (``transformers`` hands back the transpose
under a different name; matching it here would mean two conversions instead of
none.) The ``_scales`` tensors are dropped: they have no meaning once the
values are dequantized, and nothing maps them.

    python -m torchtitan.experiments.retroplanner.scripts.dequantize_gpt_oss \\
        --src  /path/to/models--openai--gpt-oss-20b/snapshots/<sha> \\
        --dest "$RP_CACHE"/gpt-oss-20b-bf16
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# e2m1: three magnitude bits, the top bit is the sign.
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def dequantize(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """[..., N, 16] uint8 + [..., N] uint8 exponents -> [..., N * 32] float32.

    Each byte holds two values, low nibble first.
    """
    lut = torch.tensor(_E2M1, dtype=torch.float32)
    packed = blocks.to(torch.int16)
    low, high = packed & 0xF, (packed >> 4) & 0xF

    def nibble(n: torch.Tensor) -> torch.Tensor:
        magnitude = lut[(n & 7).long()]
        return torch.where(n >= 8, -magnitude, magnitude)

    values = torch.stack([nibble(low), nibble(high)], dim=-1)
    values = values.reshape(*blocks.shape[:-1], 32)
    exponent = torch.pow(2.0, scales.float() - 127.0).unsqueeze(-1)
    return (values * exponent).reshape(*blocks.shape[:-2], -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    args = parser.parse_args()

    index_path = args.src / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    args.dest.mkdir(parents=True, exist_ok=True)

    # Group the work by source shard so each file is opened once.
    by_file: dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        by_file.setdefault(filename, []).append(key)

    new_map: dict[str, str] = {}
    total_bytes = 0
    for shard_index, (filename, keys) in enumerate(sorted(by_file.items())):
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(args.src / filename, framework="pt") as handle:
            names = set(handle.keys())
            for key in sorted(keys):
                if key.endswith("_scales"):
                    continue  # meaningless once dequantized; nothing maps it
                if key.endswith("_blocks"):
                    scales_key = key[: -len("_blocks")] + "_scales"
                    if scales_key not in names:
                        raise ValueError(f"{key} has no matching {scales_key}")
                    value = dequantize(
                        handle.get_tensor(key), handle.get_tensor(scales_key)
                    ).to(torch.bfloat16)
                else:
                    value = handle.get_tensor(key).to(torch.bfloat16)
                tensors[key] = value

        out_name = f"model-{shard_index:05d}.safetensors"
        save_file(tensors, str(args.dest / out_name), metadata={"format": "pt"})
        for key, value in tensors.items():
            new_map[key] = out_name
            total_bytes += value.numel() * value.element_size()
        print(f"  {out_name}: {len(tensors)} tensors", flush=True)
        del tensors

    (args.dest / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_bytes}, "weight_map": new_map}, indent=1
        )
    )

    # config.json must stop advertising mxfp4 or a reader will try to unpack
    # tensors that are already plain bf16.
    config = json.loads((args.src / "config.json").read_text())
    config.pop("quantization_config", None)
    (args.dest / "config.json").write_text(json.dumps(config, indent=2))

    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
    ):
        source = args.src / name
        if source.exists():
            shutil.copy2(source, args.dest / name)

    print(f"wrote {len(new_map)} tensors, {total_bytes / 2**30:.1f} GiB -> {args.dest}")


if __name__ == "__main__":
    main()
