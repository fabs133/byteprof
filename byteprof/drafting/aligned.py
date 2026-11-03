"""Patch-aligned round policies (pure functions, no model) and their simulator.

A round starts after the last emitted atom a_i. It is *at a patch boundary* when a_i carries the boundary flag
(the next byte starts a new patch) and *mid-patch* otherwise. The round rules are those of `sim.simulate`: byte (and,
for the crossing rule, flag) must match, one free atom per round, the cap K, at most `max_patches` patches.

Policies (TASK_patch_aligned.md):

* P0  within-patch lookup (single-patch rule, the best so far)
* P1  crossing lookup every round (crossing rule)
* P2  hybrid: the lookup drafter's completion of the current patch (up to its predicted boundary), then the LM's
      continuation recorded at that boundary; crossing rule. P2b: the same, but a round that starts at a boundary
      uses the LM directly
* P3  boundary-gated: mid-patch rounds use the lookup within the patch (single-patch rule); rounds at a boundary use
      the LM's crossing draft
* P4  boundary-gated lookup: as P3, but the lookup drafts the crossing at boundaries (no LM)
* L1  the LM drafts the crossing every round (the Task B protocol; reference for the cost-model rerun)

A drafter's continuation only matters when everything drafted before it was right, i.e. when its context equals the
true text. The LM continuation after a predicted boundary at truth position j is therefore the one recorded at j, and
`accepted_prefix` never reaches it when the completion before it is wrong (bytes or flag).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from . import flag_of
from .diagnostics import limit_patches
from .sim import accepted_prefix

POLICIES = ("P0", "P1", "P2", "P2b", "P3", "P4", "L1")
MAX_PATCHES = 8


class Sources(Protocol):
    """What the policies draw on. `i` is the truth index of the last emitted atom."""

    def lookup_raw(self, k: int) -> list[int]: ...  # within-patch lookup draft (flags are ignored by the rule)
    def lookup_flagged(self, k: int) -> list[int]: ...  # lookup draft with predicted boundary flags
    def lm_at(self, j: int, k: int) -> tuple[list[int], int]: ...  # (LM draft after truth[j], tokens it needed)
    def push(self, atoms: Sequence[int]) -> None: ...


@dataclass
class Plan:
    draft: list[int]
    multi: bool  # crossing rule for this round
    lm_call: bool = False
    lm_tokens: int = 0
    lookup_raw_call: bool = False
    lookup_flagged_call: bool = False
    completion_len: int = 0  # P2: atoms before the LM part


def completion(draft: Sequence[int]) -> tuple[list[int], bool]:
    """The draft up to and including its first flagged atom, and whether it has one."""
    for j, a in enumerate(draft):
        if flag_of(a):
            return list(draft[: j + 1]), True
    return list(draft), False


def hybrid_draft(comp: Sequence[int], lm_part: Sequence[int], k: int) -> list[int]:
    """Completion followed by the LM part, cut to K in total."""
    return (list(comp) + list(lm_part))[:k]


def oracle_completion(truth: Sequence[int], i: int, k: int) -> list[int]:
    """The true atoms from a_{i+1} through the next flagged atom (at most K)."""
    out: list[int] = []
    for a in truth[i + 1: i + 1 + k]:
        out.append(a)
        if flag_of(a):
            break
    return out


def plan_round(policy: str, i: int, truth: Sequence[int], src: Sources, k: int, *, oracle: bool = False,
               k_within: int | None = None) -> Plan:
    """The draft and rule of one round. `oracle`: the lookup parts (not the LM part) are the true atoms (upper bound).
    `k_within` (P4 only, TASK_build_p4.md): K of the mid-patch rounds; `k` is then the K of the crossing rounds."""
    if policy not in POLICIES:
        raise ValueError(policy)
    at_boundary = bool(flag_of(truth[i]))

    def lm_direct() -> Plan:
        d, m = src.lm_at(i, k)
        return Plan(list(d), True, lm_call=True, lm_tokens=m)

    if policy == "L1":
        return lm_direct()
    if policy == "P0":
        return Plan(list(src.lookup_raw(k)), False, lookup_raw_call=True)
    if policy == "P1":
        return Plan(list(src.lookup_flagged(k)), True, lookup_flagged_call=True)
    if policy in ("P2", "P2b"):
        if policy == "P2b" and at_boundary:
            return lm_direct()
        look = oracle_completion(truth, i, k) if oracle else src.lookup_flagged(k)
        comp, found = completion(look)
        plan = Plan(comp, True, lookup_flagged_call=not oracle, completion_len=len(comp))
        if found and len(comp) < k:  # the LM is asked at the predicted boundary
            lm, m = src.lm_at(i + len(comp), k - len(comp))
            plan.draft = hybrid_draft(comp, lm, k)
            plan.lm_call, plan.lm_tokens = True, m
        return plan
    if policy == "P3":
        if at_boundary:
            return lm_direct()
        look = oracle_completion(truth, i, k) if oracle else src.lookup_raw(k)
        return Plan(list(look), False, lookup_raw_call=not oracle)
    # P4
    if at_boundary:
        if oracle:
            return Plan(list(truth[i + 1: i + 1 + k]), True)
        return Plan(list(src.lookup_flagged(k)), True, lookup_flagged_call=True)
    kw = k if k_within is None else k_within
    look = oracle_completion(truth, i, kw) if oracle else src.lookup_raw(kw)
    return Plan(list(look), False, lookup_raw_call=not oracle)


@dataclass
class PolicyResult:
    policy: str
    k: int
    oracle: bool = False
    emitted: int = 0
    rounds: int = 0
    global_passes: int = 0
    by_kp: dict[tuple[int, int, bool, bool], int] = field(default_factory=dict)  # (draft len, patches, crossing, lm) -> rounds
    lm_calls: int = 0
    lookup_raw_calls: int = 0
    lookup_flagged_calls: int = 0
    lm_m_hist: dict[int, int] = field(default_factory=dict)
    kinds: dict[str, int] = field(default_factory=lambda: {"within": 0, "cross": 0})
    start_rounds: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})  # start state (flag of a_i) -> rounds
    start_bytes: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})
    trans: dict[tuple[int, int], int] = field(default_factory=dict)  # (start state, next start state) -> rounds
    accepted: int = 0
    drafted: int = 0

    @property
    def bytes_per_round(self) -> float:
        return self.emitted / self.rounds if self.rounds else 0.0

    @property
    def bytes_per_global_pass(self) -> float:
        return self.emitted / self.global_passes if self.global_passes else float("inf")

    @property
    def lm_calls_per_byte(self) -> float:
        return self.lm_calls / self.emitted if self.emitted else 0.0

    def merge(self, o: "PolicyResult") -> None:
        for f in ("emitted", "rounds", "global_passes", "lm_calls", "lookup_raw_calls", "lookup_flagged_calls", "accepted", "drafted"):
            setattr(self, f, getattr(self, f) + getattr(o, f))
        for name in ("by_kp", "lm_m_hist", "kinds", "start_rounds", "start_bytes", "trans"):
            mine, theirs = getattr(self, name), getattr(o, name)
            for key, v in theirs.items():
                mine[key] = mine.get(key, 0) + v


def simulate_policy(truth: Sequence[int], policy: str, src: Sources, k: int, *, max_new: int | None = None,
                    ks: Sequence[int] | None = None, max_patches: int = MAX_PATCHES, oracle: bool = False,
                    k_within: int | None = None) -> PolicyResult:
    """Replay a policy on the recorded truth a_0 .. a_n. `ks`: the verification sizes that exist (a draft is cut to the
    largest of them not above its length), as in `sim.simulate`."""
    n_avail = len(truth) - 1
    n = n_avail if max_new is None else min(max_new, n_avail)
    res = PolicyResult(policy, k, oracle)
    i = 0
    while res.emitted < n:
        plan = plan_round(policy, i, truth, src, k, oracle=oracle, k_within=k_within)
        draft = list(plan.draft[:k if (plan.multi or k_within is None) else k_within])
        if plan.multi:
            draft, _ = limit_patches(draft, flag_of(truth[i]), max_patches)
        if ks is not None:
            draft = draft[: max([kv for kv in ks if kv <= len(draft)], default=0)]
        preds = truth[i + 1: i + 2 + k]
        L = accepted_prefix(draft, preds, k, plan.multi)
        take = min(1 + L, n - res.emitted)
        a0_flag = flag_of(truth[i])
        if plan.multi:
            needs_global = bool(a0_flag) or any(flag_of(truth[i + j]) for j in range(1, L + 1))
            n_patches = a0_flag + sum(1 for dj in draft if flag_of(dj))
        else:
            needs_global = bool(a0_flag)
            n_patches = a0_flag
        key = (len(draft), n_patches, plan.multi, plan.lm_call)
        res.by_kp[key] = res.by_kp.get(key, 0) + 1
        res.rounds += 1
        res.global_passes += int(needs_global)
        res.kinds["cross" if plan.multi else "within"] += 1
        res.lm_calls += int(plan.lm_call)
        res.lookup_raw_calls += int(plan.lookup_raw_call)
        res.lookup_flagged_calls += int(plan.lookup_flagged_call)
        if plan.lm_call:
            res.lm_m_hist[plan.lm_tokens] = res.lm_m_hist.get(plan.lm_tokens, 0) + 1
        res.start_rounds[a0_flag] += 1
        res.start_bytes[a0_flag] += take
        nxt = flag_of(truth[i + take])
        res.trans[(a0_flag, nxt)] = res.trans.get((a0_flag, nxt), 0) + 1
        res.accepted += L
        res.drafted += len(draft)
        src.push(truth[i + 1: i + 1 + take])
        res.emitted += take
        i += take
    return res


# --- two-state model ------------------------------------------------------------------------------------------

def two_state_bytes_per_round(trans: dict[tuple[int, int], float], bytes_per_round: dict[int, float]) -> float:
    """Bytes per round of a Markov chain over the restart state (0 = mid-patch/mid-token, 1 = at a boundary):
    `trans[(s, t)]` = rounds that started in s and were followed by a round that starts in t (counts or
    probabilities), `bytes_per_round[s]` = mean bytes emitted by a round that starts in s. Returns the
    stationary-weighted mean."""
    p = {}
    for s in (0, 1):
        tot = trans.get((s, 0), 0.0) + trans.get((s, 1), 0.0)
        for t in (0, 1):
            p[(s, t)] = trans.get((s, t), 0.0) / tot if tot else (1.0 if s == t else 0.0)
    a, b = p[(0, 1)], p[(1, 0)]  # P(mid -> boundary), P(boundary -> mid)
    if a + b == 0:  # no switching: the chain stays where it starts; weight both states equally
        pi1 = 0.5
    else:
        pi1 = a / (a + b)
    return (1 - pi1) * bytes_per_round[0] + pi1 * bytes_per_round[1]


def two_state_figure(h_boundary: float, h_mid: float, q_boundary: float, free_boundary: float, free_mid: float,
                     k: int = 32) -> float:
    """Bytes per round of the analytic two-state model behind the paper figure.

    A round that starts in state s emits `free_s` (the free unit: 1 byte, or about 4 for one token) plus a geometric
    number of accepted bytes with per-byte hit rate h_s, capped at K. The next round starts at a boundary with
    probability `q_boundary` (1 for a token free unit, which always ends on a token end), otherwise mid-token.
    The chain has no memory of the previous state, so the share of boundary rounds is `q_boundary`."""
    def e(h: float, free: float) -> float:
        h = min(max(h, 0.0), 1.0)
        acc = k if h >= 1.0 else min(h * (1 - h ** k) / (1 - h), k)
        return free + acc

    return q_boundary * e(h_boundary, free_boundary) + (1 - q_boundary) * e(h_mid, free_mid)


# --- concrete sources over recorded data ------------------------------------------------------------------------

class RecordedSources:
    """`Sources` over a lookup drafter (e.g. the combined drafter), recorded LM continuations (item j = the
    continuation after truth[j], `lmdraft.Cont`) and the dolma2 tokenizer (flags of lookup drafts and LM drafts).
    `lookup_flagged` gives the lookup draft the dolma2 boundary flags of the retokenized context + draft (the `cut`
    protocol of the crossing task)."""

    def __init__(self, lookup, dol, ctx_atoms: Sequence[int], truth: Sequence[int], lm_items=None, flag_mode: str = "cut"):
        self.lookup, self.dol, self.flag_mode = lookup, dol, flag_mode
        self.prompt, self.items = bytes(a & 255 for a in ctx_atoms), lm_items
        self.tb = bytes(a & 255 for a in truth)
        self.i = 0
        lookup.reset(list(ctx_atoms) + [truth[0]], truth)
        self.lm_times: list[float] = []

    def _ctx(self, j: int) -> bytes:
        return self.prompt + self.tb[: j + 1]

    def lookup_raw(self, k: int) -> list[int]:
        return list(self.lookup.propose(k))

    def lookup_flagged(self, k: int) -> list[int]:
        from .lmdraft import Cont, build_draft

        d = self.lookup.propose(k)
        if not d:
            return []
        cont = bytes(a & 255 for a in d)
        return build_draft(self._ctx(self.i), Cont(0, cont, [len(cont)]), k, self.dol, self.flag_mode)[0]

    def lm_at(self, j: int, k: int) -> tuple[list[int], int]:
        import time

        from .lmdraft import build_draft

        if self.items is None or j >= len(self.items) or k <= 0:
            return [], 0
        t = time.perf_counter()
        atoms, m = build_draft(self._ctx(j), self.items[j], k, self.dol, self.flag_mode)
        self.lm_times.append(time.perf_counter() - t)
        return atoms, m

    def push(self, atoms: Sequence[int]) -> None:
        self.lookup.push(atoms)
        self.i += len(atoms)
