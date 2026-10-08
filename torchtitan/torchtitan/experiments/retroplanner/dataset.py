# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Packed multi-turn dataloader for retrosynthesis agent trajectories.

``torchtitan``'s ``ChatDataset`` handles a single ``[user, assistant]`` pair.
These records are whole episodes, and every assistant turn in them is a
supervision target, so this dataset does its own harmony rendering and carries
two extra per-token streams (role id, decision span id) through packing.

Batches are ``({"input", "positions", "role_ids", "span_ids"}, labels)``.
``positions`` restarts at 0 on each trajectory, which is what the varlen/flex
attention backends use to keep packed trajectories from attending to each
other; ``role_ids``/``span_ids`` are consumed by the loss, not by the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import torch
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.tools.logging import logger

from .harmony import HarmonyTrajectoryEncoder, NO_SPAN
from .trajectory import ROLE_IGNORE, to_trajectory


@dataclass
class PackingStats:
    """Counters describing what the packer did to the corpus.

    Reported once per epoch. ``dropped_too_long`` is the number that matters:
    a dropped trajectory is silently absent from training.
    """

    trajectories_packed: int = 0
    trajectories_dropped: int = 0
    decisions_packed: int = 0
    tokens_emitted: int = 0
    tokens_supervised: int = 0
    tokens_padding: int = 0

    def as_dict(self) -> dict[str, float]:
        emitted = max(self.tokens_emitted, 1)
        seen = max(self.trajectories_packed + self.trajectories_dropped, 1)
        return {
            "data/trajectories_packed": self.trajectories_packed,
            "data/trajectories_dropped": self.trajectories_dropped,
            "data/dropped_frac": self.trajectories_dropped / seen,
            "data/decisions_packed": self.decisions_packed,
            "data/supervised_frac": self.tokens_supervised / emitted,
            "data/padding_frac": self.tokens_padding / emitted,
        }


class RetroTrajectoryDataset(IterableDataset, Stateful):
    """Greedy-packed harmony encoding of whole agent trajectories."""

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: BaseTokenizer,
        *,
        seq_len: int,
        current_date: str,
        reasoning_effort: str = "medium",
        max_spans_per_sequence: int = 256,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        seed: int = 42,
    ) -> None:
        self._encoder = HarmonyTrajectoryEncoder(
            tokenizer,
            current_date=current_date,
            reasoning_effort=reasoning_effort,
        )
        self._eos_id = self._encoder.eos_id
        self.seq_len = seq_len
        self.infinite = infinite
        self.max_spans_per_sequence = max_spans_per_sequence
        self._seed = seed

        self._original_data = split_dataset_by_node(
            cast(Dataset, dataset.shuffle(seed=seed)), dp_rank, dp_world_size
        )
        self._data = self._original_data
        self._dataset_id = f"{dataset.info.dataset_name}/{dataset.split}"

        self._sample_idx = 0
        self._epoch = 0
        self.stats = PackingStats()

        self._inputs: list[int] = []
        self._labels: list[int] = []
        self._roles: list[int] = []
        self._spans: list[int] = []
        self._positions: list[int] = []
        self._next_span_id = 0
        self._pending: dict[str, list[int]] | None = None
        self._logged_first_sample = False

    def _get_data_iter(self):
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            return iter(self._data.skip(self._sample_idx))
        return iter(self._data)

    def _encode_sample(self, record: dict[str, Any]) -> dict[str, list[int]] | None:
        """Encode one record, or None if it cannot fit in a single sequence."""
        encoded = self._encoder.encode(to_trajectory(record), first_span_id=0)
        if len(encoded) > self.seq_len:
            logger.debug(
                "Dropping trajectory route_idx=%s: %d tokens exceeds seq_len %d",
                record.get("route_idx"),
                len(encoded),
                self.seq_len,
            )
            return None
        if encoded.num_decisions > self.max_spans_per_sequence:
            raise ValueError(
                f"Trajectory route_idx={record.get('route_idx')} has "
                f"{encoded.num_decisions} decisions, above max_spans_per_sequence="
                f"{self.max_spans_per_sequence}. Raise the limit."
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

    def _log_first_sample(self, encoded) -> None:
        supervised = sum(1 for r in encoded.roles if r != ROLE_IGNORE)
        logger.info(
            "[RetroTrajectoryDataset] first sample: %d tokens, %d supervised "
            "(%.1f%%), %d decisions",
            len(encoded),
            supervised,
            100.0 * supervised / max(len(encoded), 1),
            encoded.num_decisions,
        )

    def __iter__(self):
        while True:
            if self._pending is not None:
                sample, self._pending = self._pending, None
                self._append(sample)
                self._sample_idx += 1
                if len(self._inputs) == self.seq_len:
                    yield self._flush()

            for record in self._get_data_iter():
                sample = self._encode_sample(cast(dict, record))
                if sample is None:
                    self._sample_idx += 1
                    self.stats.trajectories_dropped += 1
                    continue

                remaining = self.seq_len - len(self._inputs)
                if len(sample["input"]) > remaining and self._inputs:
                    self._pad(remaining)
                    self._pending = sample
                    yield self._flush()
                    sample, self._pending = self._pending, None

                self._append(sample)
                self._sample_idx += 1
                if len(self._inputs) == self.seq_len:
                    yield self._flush()

            if self._inputs:
                self._pad(self.seq_len - len(self._inputs))
                yield self._flush()

            logger.info(
                "[RetroTrajectoryDataset] epoch %d packing stats: %s",
                self._epoch,
                self.stats.as_dict(),
            )
            if not self.infinite:
                logger.warning("Dataset '%s' has run out of data", self._dataset_id)
                return
            self.stats = PackingStats()
            self._reloop()

    def _append(self, sample: dict[str, list[int]]) -> None:
        n = len(sample["input"])
        self._inputs.extend(sample["input"])
        self._labels.extend(sample["labels"])
        self._roles.extend(sample["roles"])
        # Renumber decision spans so ids are unique within the packed sequence.
        base = self._next_span_id
        self._spans.extend(
            NO_SPAN if s == NO_SPAN else base + s for s in sample["spans"]
        )
        num_decisions = max((s for s in sample["spans"]), default=NO_SPAN) + 1
        self._next_span_id += num_decisions
        self._positions.extend(range(n))

        self.stats.trajectories_packed += 1
        self.stats.decisions_packed += num_decisions
        self.stats.tokens_emitted += n
        self.stats.tokens_supervised += sum(
            1 for r in sample["roles"] if r != ROLE_IGNORE
        )

    def _pad(self, pad_len: int) -> None:
        if pad_len <= 0:
            return
        self._inputs.extend([self._eos_id] * pad_len)
        self._labels.extend([IGNORE_INDEX] * pad_len)
        self._roles.extend([ROLE_IGNORE] * pad_len)
        self._spans.extend([NO_SPAN] * pad_len)
        self._positions.extend(range(pad_len))
        self.stats.tokens_emitted += pad_len
        self.stats.tokens_padding += pad_len

    def _flush(self):
        if self._next_span_id > self.max_spans_per_sequence:
            raise ValueError(
                f"Packed sequence holds {self._next_span_id} decisions, above "
                f"max_spans_per_sequence={self.max_spans_per_sequence}."
            )
        batch = (
            {
                "input": torch.tensor(self._inputs, dtype=torch.long),
                "positions": torch.tensor(self._positions, dtype=torch.long),
                "role_ids": torch.tensor(self._roles, dtype=torch.long),
                "span_ids": torch.tensor(self._spans, dtype=torch.long),
            },
            torch.tensor(self._labels, dtype=torch.long),
        )
        self._inputs, self._labels = [], []
        self._roles, self._spans, self._positions = [], [], []
        self._next_span_id = 0
        return batch

    def _reloop(self) -> None:
        self._sample_idx = 0
        self._epoch += 1
        if isinstance(self._data, Dataset):
            self._data = cast(
                Dataset, self._original_data.shuffle(seed=self._seed + self._epoch)
            )
        elif hasattr(self._data, "set_epoch"):
            self._data.set_epoch(self._epoch)
        logger.warning(
            "Dataset '%s' is being re-looped (epoch %d)", self._dataset_id, self._epoch
        )

    def state_dict(self):
        state: dict[str, Any] = {
            "epoch": self._epoch,
            "inputs": self._inputs,
            "labels": self._labels,
            "roles": self._roles,
            "spans": self._spans,
            "positions": self._positions,
            "next_span_id": self._next_span_id,
            "pending": self._pending,
        }
        if isinstance(self._data, Dataset):
            state["sample_idx"] = self._sample_idx
        else:
            state["data"] = self._data.state_dict()
        return state

    def load_state_dict(self, state_dict):
        self._epoch = state_dict["epoch"]
        self._inputs = state_dict["inputs"]
        self._labels = state_dict["labels"]
        self._roles = state_dict["roles"]
        self._spans = state_dict["spans"]
        self._positions = state_dict["positions"]
        self._next_span_id = state_dict["next_span_id"]
        self._pending = state_dict["pending"]
        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
            if self._epoch > 0:
                self._data = cast(
                    Dataset, self._original_data.shuffle(seed=self._seed + self._epoch)
                )
        else:
            data_state = state_dict["data"]
            self._data.set_epoch(data_state.get("epoch", 0))
            self._data.load_state_dict(data_state)


class RetroTrajectoryDataLoader(ParallelAwareDataloader):
    """Dataloader over a local jsonl of retrosynthesis agent trajectories."""

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset_path: str | None = None
        """Path to the trajectory jsonl (conv_*.oss.jsonl). Required."""

        split: str = "train"
        """Split name passed to ``load_dataset('json', ...)``."""

        current_date: str = "2025-06-01"
        """Pinned "Current date" of the harmony system message."""

        reasoning_effort: str = "medium"
        """Harmony reasoning-effort field of the system message."""

        max_spans_per_sequence: int = 256
        """Capacity of the per-sequence decision-span accounting buffers."""

        infinite: bool = True
        """Loop the corpus. Multi-GPU runs need this to avoid a ragged epoch."""

        seed: int = 42

        load_dataset_kwargs: dict[str, Any] = field(default_factory=dict)
        """Extra kwargs forwarded to ``datasets.load_dataset``."""

        def __post_init__(self) -> None:
            if not self.dataset_path:
                raise ValueError(
                    "RetroTrajectoryDataLoader.Config requires dataset_path "
                    "(the conv_*.oss.jsonl produced by prepare_dataset.py)."
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
        ds = RetroTrajectoryDataset(
            cast(Dataset, dataset),
            tokenizer,
            seq_len=seq_len,
            current_date=config.current_date,
            reasoning_effort=config.reasoning_effort,
            max_spans_per_sequence=config.max_spans_per_sequence,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
            seed=config.seed,
        )
        super().__init__(
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
