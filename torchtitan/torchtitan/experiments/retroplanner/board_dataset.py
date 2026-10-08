# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Packing and loading for the board format.

Reuses :class:`dataset.RetroTrajectoryDataset` wholesale -- greedy packing,
padding, span renumbering, epoch stats and the Stateful protocol are all format
agnostic.  Only two things change: the record converter and the encoder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from datasets import Dataset, load_dataset

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.tools.logging import logger

from .board_harmony import BoardHarmonyEncoder
from .board_qwen import BoardQwenEncoder
from .board_trajectory import to_board_trajectory
from .dataset import RetroTrajectoryDataLoader, RetroTrajectoryDataset

# Which wire format BoardTrajectoryDataLoader.Config.encoder selects. Both
# classes share the same (tokenizer, *, supervise_terminate) constructor and
# encode(traj, *, first_span_id) -> EncodedTrajectory contract, so the dataset
# and dataloader below need no per-format branching beyond this lookup.
BOARD_ENCODERS: dict[str, type] = {
    "harmony": BoardHarmonyEncoder,
    "qwen": BoardQwenEncoder,
}


class BoardTrajectoryDataset(RetroTrajectoryDataset):
    """Greedy-packed board episodes.

    The system and developer preambles are lifted from each record's own
    `raw_text`, so `current_date` and `reasoning_effort` are NOT re-derived here:
    whatever the builder pinned is what the model trains on, and the same string
    is what a serving stack reproduces from the same tools declaration.
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: BaseTokenizer,
        *,
        encoder_cls: type = BoardHarmonyEncoder,
        supervise_terminate: bool = True,
        **kwargs,
    ) -> None:
        kwargs.setdefault("current_date", "unused")
        super().__init__(dataset, tokenizer, **kwargs)
        self._encoder = encoder_cls(tokenizer, supervise_terminate=supervise_terminate)
        self._eos_id = self._encoder.eos_id

    def _encode_sample(self, record: dict[str, Any]) -> dict[str, list[int]] | None:
        encoded = self._encoder.encode(to_board_trajectory(record), first_span_id=0)
        if len(encoded) > self.seq_len:
            logger.debug(
                "Dropping episode target=%.24s route=%s: %d tokens exceeds "
                "seq_len %d",
                record.get("target", ""),
                (record.get("route") or {}).get("set_hash"),
                len(encoded),
                self.seq_len,
            )
            return None
        if encoded.num_decisions > self.max_spans_per_sequence:
            raise ValueError(
                f"episode has {encoded.num_decisions} calls, above "
                f"max_spans_per_sequence={self.max_spans_per_sequence}"
            )
        if not self._logged_first_sample:
            self._log_first_sample(encoded)
            self._logged_first_sample = True
        return {
            "input": encoded.input_ids,
            "labels": encoded.labels,
            "roles": encoded.roles,
            "spans": encoded.span_ids,
        }


class BoardTrajectoryDataLoader(RetroTrajectoryDataLoader):
    """Dataloader over a local jsonl of board episodes."""

    @dataclass(kw_only=True, slots=True)
    class Config(RetroTrajectoryDataLoader.Config):
        supervise_terminate: bool = True
        """Put the closing stop message in the loss (as its own role).  Turn it
        off to train the tool calls alone -- it is the longest target in
        an episode and the only one whose content is not checkable."""

        encoder: str = "harmony"
        """Wire format to encode board episodes into: "harmony" (gpt-oss
        control tokens, see board_harmony.py) or "qwen" (Qwen3.5 chat
        template -- <think>/<tool_call>, see board_qwen.py)."""

        def __post_init__(self) -> None:
            # Explicit two-argument super(): a slots=True dataclass rebuilds the
            # class object after __post_init__ is defined, so the zero-argument
            # super() magic (which closes over __class__ at definition time)
            # resolves against the pre-slots class and raises "obj must be an
            # instance or subtype of type".
            super(BoardTrajectoryDataLoader.Config, self).__post_init__()
            if self.encoder not in BOARD_ENCODERS:
                raise ValueError(
                    f"unknown board encoder {self.encoder!r}; choose one of "
                    f"{sorted(BOARD_ENCODERS)}"
                )

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        snapshot_every_n_steps: int | None = 1,
        **kwargs,
    ):
        dataset = load_dataset(
            "json",
            data_files=config.dataset_path,
            split=config.split,
            **config.load_dataset_kwargs,
        )
        ds = BoardTrajectoryDataset(
            cast(Dataset, dataset),
            tokenizer,
            encoder_cls=BOARD_ENCODERS[config.encoder],
            seq_len=seq_len,
            supervise_terminate=config.supervise_terminate,
            max_spans_per_sequence=config.max_spans_per_sequence,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
            seed=config.seed,
        )
        # Skip RetroTrajectoryDataLoader.__init__ (it builds the other dataset)
        # and go straight to the parallel-aware base with ours.
        from torchtitan.components.dataloader import ParallelAwareDataloader

        ParallelAwareDataloader.__init__(
            self,
            ds,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            snapshot_every_n_steps=snapshot_every_n_steps,
            batch_size=local_batch_size,
        )
