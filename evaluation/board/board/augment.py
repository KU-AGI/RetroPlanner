#!/usr/bin/env python
"""Several training instances per target, from the one set of recorded routes.

One episode per target that realises every selected route is the right SHAPE --
it is what puts route diversity inside a single rollout instead of across ten
samples.  It is the wrong CORPUS size: one instance per target, each target
seen in exactly one board state, one candidate numbering, one claiming order.
What a model can learn from that is the trajectory, not the environment: the
agent reproduces one remembered walk and has nothing to fall back on when the
board does not match it.

So: keep the shape, vary everything about the presentation that the ANSWER does
not depend on.  Five axes, and each one is a different claim about what the model
must not be reading:

  perm     the menu is renumbered.  The single-step model's own order becomes one
           arbitrary ordering among many, so `c0` stops being a synonym for "the
           one to take".  This is the strongest of the five and the only one that
           changes the OBSERVATION while provably leaving the decision alone: the
           DP ranks on the axis and the reactants, neither of which moves, so the
           labelled ranking names the same disconnections in the same order and
           only the numbers differ.  The single-step model's c0 is often not
           the DP's own argmax, so a corpus in the model's order teaches a
           positional prior that is frequently wrong.

  order    the queue is claimed in a different order.  Which route is built first
           decides what is already on the board when the second is claimed, so
           the same set of routes gives genuinely different boards and different
           `done` payloads -- the shared-subtree lesson, from both directions.

  cluster  the queue is one CLUSTER of routes that share most of their steps.
           A cluster is the sub-unit the board is cheap on ("the second route is
           mostly the first route's graph"), and isolating it teaches that
           without the noise of an unrelated route in the same episode.

  trap     the rollout departs from the DP once, walks into a failing subtree,
           and the recovery -- `dead` with a reason, then the correct re-rank --
           is labelled.  Route-following alone never produces a dead end, so
           without this the corpus contains no example of backing out.

  free     a short queue plus a budget for routes the graph ALREADY holds.  An
           all-purchasable candidate closes its molecule the moment it is
           applied, so swapping one into a claimed route is another complete
           route for one action.  This is the cheapest route count there is, and
           route-following alone rarely demonstrates it.

Nothing here re-runs a tool or invents chemistry.  Every variant is the same
recorded menus and the same selected routes, presented differently.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Optional

from .episode import DataWorld, GraphLabels
from .state import Candidate

KINDS = ("all", "perm", "order", "cluster", "single", "trap", "free")


# ------------------------------------------------------------------ the menu
def _perm(seed: int, smiles: str, n: int, window: Optional[int]) -> list[int]:
    """A deterministic permutation of a molecule's menu, stable within an episode.

    Deterministic because the menu is rebuilt on every lookup -- `DP.value`
    recurses over `world.menu()` thousands of times and `_do_open` calls it again
    to freeze what is on screen.  A fresh shuffle per call would give the DP one
    numbering and the board another, and the labelled ranking would name
    candidates the model was never shown.

    Permuting only the first `window` entries among themselves is deliberate:
    those are the ones on screen, and a permutation that moved a candidate across
    the fold would push a route's own step out of the window and drop the
    instance as off-route.  The tail keeps its order because nothing can name it.
    """
    idx = list(range(n))
    end = n if window is None else min(window, n)
    if end < 2:
        return idx
    # blake2b, not hash(): str hashing is salted per process, so hash() would
    # give a different corpus on every run and nothing would be reproducible.
    h = hashlib.blake2b(f"{seed}\x00{smiles}".encode(), digest_size=16).digest()
    head = idx[:end]
    # Fisher-Yates driven by the digest, extended by rehashing when it runs out.
    stream, k = list(h), 0
    for i in range(end - 1, 0, -1):
        if k + 2 > len(stream):
            h = hashlib.blake2b(bytes(stream) + b"\x01", digest_size=16).digest()
            stream, k = list(h), 0
        j = ((stream[k] << 8) | stream[k + 1]) % (i + 1)
        k += 2
        head[i], head[j] = head[j], head[i]
    return head + idx[end:]


@dataclass
class PermutedWorld(DataWorld):
    """A DataWorld that renumbers each menu.  Same chemistry, different c-numbers."""

    perm_seed: int = 0
    perm_window: Optional[int] = None

    def menu(self, smiles: str) -> list[Candidate]:
        rows = self.menus.get(smiles, [])[: self.top_k]
        if not rows:
            return []
        order = _perm(self.perm_seed, smiles, len(rows), self.perm_window)
        out = []
        for i, j in enumerate(order):
            r, q = rows[j]
            sig = ({"q": q} if self.scores is None
                   else self.scores.signals(smiles, list(r), q))
            out.append(Candidate(i, list(r), sig))
        return out


def permuted_world(world: DataWorld, seed: int,
                   window: Optional[int] = None) -> PermutedWorld:
    return PermutedWorld(menus=world.menus, stock=world.stock, prices=world.prices,
                         scope=world.scope, top_k=world.top_k, scores=world.scores,
                         perm_seed=seed, perm_window=window)


def permuted_labels(labels: Optional[GraphLabels], world: DataWorld, seed: int,
                    window: Optional[int]) -> Optional[GraphLabels]:
    """Move the recorded used/dead labels onto the new c-numbers.

    The labels join to the menu POSITIONALLY (graph candidate rank r is menu index
    r-1).  Renumbering the menu without renumbering these would hand the trap
    picker the label of a different candidate -- which is not a weaker signal, it
    is a wrong one, and it would put `dead` on a branch that closes.
    """
    if labels is None:
        return None
    out = GraphLabels(has_solution=dict(labels.has_solution),
                      join_ok=labels.join_ok, join_bad=labels.join_bad)
    by_mol: dict[str, dict[int, str]] = {}
    for (smi, idx), lab in labels.label.items():
        by_mol.setdefault(smi, {})[idx] = lab
    for smi, per in by_mol.items():
        n = len(world.menus.get(smi, [])[: world.top_k])
        if not n:
            continue
        order = _perm(seed, smi, n, window)
        for new_i, old_i in enumerate(order):
            lab = per.get(old_i)
            if lab is not None:
                out.label[(smi, new_i)] = lab
    return out


# ------------------------------------------------------------- route clusters
def _steps_key(route: dict) -> frozenset:
    return frozenset(f"{p}>>{'.'.join(sorted(rs))}" for p, rs in route["steps"])


def cluster_routes(routes: list[dict], threshold: float = 0.5) -> list[list[dict]]:
    """Group routes that share most of their steps.  Greedy, seeded by the best.

    Jaccard over the step set, not over the leaves: two routes can end on the same
    shelf and diverge completely in the middle, and it is the shared MIDDLE that
    makes the second claim cheap.  Greedy single-pass rather than a proper
    clustering because the input is a handful of routes already sorted by pareto
    rank, and the seed order is the thing worth preserving.
    """
    keys = [_steps_key(r) for r in routes]
    used = [False] * len(routes)
    out: list[list[dict]] = []
    for i, r in enumerate(routes):
        if used[i]:
            continue
        used[i] = True
        group, gk = [r], keys[i]
        for j in range(i + 1, len(routes)):
            if used[j]:
                continue
            inter = len(gk & keys[j])
            union = len(gk | keys[j]) or 1
            if inter / union >= threshold:
                used[j] = True
                group.append(routes[j])
                gk = gk | keys[j]
        out.append(group)
    return out


# ------------------------------------------------------------------- variants
@dataclass
class Variant:
    """One instance to build: which routes, in what order, on which board."""

    kind: str
    queue: list[dict]
    perm_seed: Optional[int] = None
    mistakes: int = 0
    route_mistakes: bool = False
    trap_depth: Optional[int] = None
    mistake_mode: Optional[str] = None
    free_routes: int = 0
    note: str = ""
    extra: dict = field(default_factory=dict)

    def meta(self) -> dict:
        return {"kind": self.kind, "perm_seed": self.perm_seed,
                "mistakes": self.mistakes, "route_mistakes": self.route_mistakes,
                "trap_depth": self.trap_depth,
                "mistake_mode": self.mistake_mode,
                "free_routes": self.free_routes,
                "n_queue": len(self.queue), "note": self.note,
                "queue": [q.get("set_hash") for q in self.queue],
                **self.extra}


def _rng(seed: int, target: str):
    import random
    h = hashlib.blake2b(f"{seed}\x00{target}".encode(), digest_size=8).digest()
    return random.Random(int.from_bytes(h, "big"))


def plan(target: str, routes: list[dict], kinds: tuple[str, ...],
         per_target: int, seed: int, queue_max: int = 8,
         cluster_threshold: float = 0.5, free_budget: int = 6,
         mistakes: int = 1, trap_depth: Optional[int] = None,
         mistake_mode: str = "readable") -> list[Variant]:
    """The variants to build for one target, capped at `per_target`.

    `all` is emitted first and always: it is the shape the corpus is built on and
    every other variant is a perturbation of it, so a run that drops it would be
    training only on perturbations of something absent.
    """
    routes = routes[:queue_max]
    if not routes:
        return []
    rng = _rng(seed, target)
    out: list[Variant] = []

    if "all" in kinds:
        out.append(Variant("all", list(routes), note="every selected route, model order"))

    if "perm" in kinds:
        out.append(Variant("perm", list(routes), perm_seed=rng.randrange(1 << 30),
                           note="every route, menu renumbered"))

    if "order" in kinds and len(routes) > 1:
        shuffled = list(routes)
        rng.shuffle(shuffled)
        if shuffled != routes:
            out.append(Variant("order", shuffled,
                               note="every route, claimed in a different order"))

    if "cluster" in kinds:
        groups = cluster_routes(routes, cluster_threshold)
        # A single group means the clustering found nothing to separate; emitting
        # it would be `all` under another name.
        if len(groups) > 1:
            for gi, g in enumerate(groups):
                out.append(Variant("cluster", g,
                                   perm_seed=rng.randrange(1 << 30) if "perm" in kinds else None,
                                   note=f"cluster {gi + 1} of {len(groups)}: "
                                        f"{len(g)} routes sharing most of their steps",
                                   extra={"cluster": gi, "n_clusters": len(groups)}))

    if "single" in kinds:
        for r in routes[: max(per_target // 2, 1)]:
            out.append(Variant("single", [r],
                               perm_seed=rng.randrange(1 << 30) if "perm" in kinds else None,
                               note="one route alone"))

    if "trap" in kinds:
        out.append(Variant("trap", list(routes), mistakes=mistakes,
                           route_mistakes=True, trap_depth=trap_depth,
                           mistake_mode=mistake_mode,
                           perm_seed=rng.randrange(1 << 30) if "perm" in kinds else None,
                           note=f"{mistakes} departure(s) from the DP, then the recovery"))

    if "free" in kinds:
        short = routes[: max(1, len(routes) // 2)]
        out.append(Variant("free", short, free_routes=free_budget,
                           perm_seed=rng.randrange(1 << 30) if "perm" in kinds else None,
                           note=f"{len(short)} queued, then claim up to {free_budget} "
                                f"routes the graph already holds"))

    # `all` first, then whatever the budget reaches, in the order declared above.
    return out[:per_target] if per_target else out
