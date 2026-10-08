# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shims for running torchtitan main against an older PyTorch.

A host whose NVIDIA driver cannot run the CUDA-13 wheels torchtitan main is
developed against has to use a CUDA-12 torch nightly. The API gap is small, and
this module closes it from inside the experiment rather than editing core.

Call ``apply_compat_shims()`` before building a config.
"""

from __future__ import annotations

import inspect

from torchtitan.tools.logging import logger

_applied = False


def _patch_create_block_mask_kwargs() -> None:
    """Drop ``separate_full_blocks`` when the installed torch does not take it.

    ``Decoder._create_flex_attention_mask`` passes
    ``separate_full_blocks=not is_in_batch_invariant_mode()``. The flag only
    chooses whether the FlexAttention kernel walks fully-unmasked blocks before
    partial ones; it does not change the mask or the attention output, and
    older torch always separates them. Dropping it therefore costs nothing
    outside batch-invariant mode, which pretraining does not use.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    import torchtitan.models.common.attention as attention

    if "separate_full_blocks" in inspect.signature(create_block_mask).parameters:
        return

    inner = attention.create_attention_mask

    def create_attention_mask(*args, **kwargs):
        kwargs.pop("separate_full_blocks", None)
        return inner(*args, **kwargs)

    attention.create_attention_mask = create_attention_mask
    # Decoder imported the symbol directly, so rebind it there too.
    import torchtitan.models.common.decoder as decoder

    if hasattr(decoder, "create_attention_mask"):
        decoder.create_attention_mask = create_attention_mask

    logger.warning(
        "torch %s has no create_block_mask(separate_full_blocks=...); dropping "
        "the argument. This only affects FlexAttention block iteration order, "
        "not the mask.",
        __import__("torch").__version__,
    )


def _patch_distributed_set_timeout() -> None:
    """Alias ``torch.distributed.set_timeout`` to its private predecessor.

    ``set_timeout`` was promoted out of ``distributed_c10d._set_pg_timeout``
    after torch 2.12. torchtitan calls the public name at the end of training
    (``set_pg_timeouts``), so without this the run dies after the last step,
    having done all the work.
    """
    import torch.distributed as dist

    if hasattr(dist, "set_timeout"):
        return

    from torch.distributed.distributed_c10d import _set_pg_timeout

    # pyrefly: ignore[missing-attribute]
    dist.set_timeout = _set_pg_timeout
    logger.warning(
        "torch.distributed.set_timeout is missing on this torch; aliasing it "
        "to the private _set_pg_timeout."
    )


def apply_compat_shims() -> None:
    """Idempotently install every shim this environment needs."""
    global _applied
    if _applied:
        return
    _patch_create_block_mask_kwargs()
    _patch_distributed_set_timeout()
    _applied = True
