"""Round-rule simulator (pure function, no model).

`truth` is the greedy atom sequence a_0 .. a_n produced by the model: a_0 is the atom the prefill
emits for the last prompt byte (the first round starts after it) and a_1 .. a_n are the n new bytes.
Because decoding is greedy and deterministic, the model's prediction at every position of a round
is `truth[i + j]` as long as the drafted prefix matched, which is all the acceptance rule needs.

Round rule (TASK_drafting.md): after the last emitted atom a0 = truth[i], a drafter proposes up to K
atoms d_1 .. d_K. The verification pass yields p_1 .. p_{K+1} = truth[i+1 .. i+K+1]. p_1 is always
emitted. d_j is accepted (so that p_{j+1} is valid) iff d_j.byte == byte(p_j) and p_j carries no
boundary flag (single-patch variant), or additionally d_j.flag == flag(p_j) and boundaries do not
end the round (multi-patch variant). Emitted per round = 1 + L, truncated at max_new.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from . import byte_of, flag_of


class Drafter(Protocol):
    def reset(self, ctx_atoms: Sequence[int], truth: Sequence[int] | None = None) -> None: ...
    def propose(self, k: int) -> list[int]: ...
    def push(self, atoms: Sequence[int]) -> None: ...


@dataclass
class SimResult:
    k: int
    multi_patch: bool
    emitted: int = 0  # new atoms emitted (a_1 ..)
    rounds: int = 0  # local passes (A_K + C_K)
    global_passes: int = 0
    l_hist: dict[int, int] = field(default_factory=dict)  # accepted drafts L -> rounds
    by_k: dict[int, int] = field(default_factory=dict)  # verification size K actually used -> rounds
    patch_rounds: dict[int, int] = field(default_factory=dict)  # K -> rounds that also ran the global model
    by_kp: dict[tuple[int, int], int] = field(default_factory=dict)  # (K, patches the round runs through the global model)
    drafted: int = 0  # drafted atoms offered over all rounds

    @property
    def bytes_per_local_pass(self) -> float:
        return self.emitted / self.rounds if self.rounds else 0.0

    @property
    def bytes_per_global_pass(self) -> float:
        return self.emitted / self.global_passes if self.global_passes else float("inf")

    @property
    def accepted(self) -> int:
        return sum(l * n for l, n in self.l_hist.items())


def accepted_prefix(draft: Sequence[int], preds: Sequence[int], k: int, multi_patch: bool) -> int:
    """L: longest prefix of the draft accepted against the predictions p_1 .. (preds[j-1] = p_j)."""
    L = 0
    for j in range(min(k, len(draft), len(preds))):
        d, p = draft[j], preds[j]
        if byte_of(d) != byte_of(p):
            break
        if multi_patch:
            if flag_of(d) != flag_of(p):
                break
        elif flag_of(p):
            break
        L += 1
    return L


def simulate(truth: Sequence[int], drafter: Drafter | None, k: int, *, ctx_atoms: Sequence[int] = (),
             max_new: int | None = None, multi_patch: bool = False, ks: Sequence[int] | None = None,
             max_patches: int | None = None) -> SimResult:
    """`ks`: the verification sizes that exist (graphs); a draft is cut to the largest K in `ks` that is not
    above its length and not above `k`, as the real loop does (`draft_core.choose_k`). None = any size."""
    n_avail = len(truth) - 1
    n = n_avail if max_new is None else min(max_new, n_avail)
    res = SimResult(k=k, multi_patch=multi_patch)
    if drafter is not None:
        drafter.reset(list(ctx_atoms) + [truth[0]], truth)
    i = 0  # index of the last emitted atom a0
    while res.emitted < n:
        draft = drafter.propose(k) if (drafter is not None and k > 0) else []
        draft = list(draft[:k])
        if multi_patch and max_patches is not None:  # a draft may carry at most `max_patches` patches (incl. a0's)
            n_p = flag_of(truth[i])
            for j, dj in enumerate(draft):
                if flag_of(dj):
                    if n_p + 1 > max_patches:
                        draft = draft[:j]
                        break
                    n_p += 1
        if ks is not None:
            draft = draft[:max([kv for kv in ks if kv <= len(draft)], default=0)]
        preds = truth[i + 1: i + 2 + k]  # p_1 .. p_{K+1}
        L = accepted_prefix(draft, preds, k, multi_patch)
        take = min(1 + L, n - res.emitted)
        # the global model runs when a0 is flagged, or (multi-patch) when an accepted draft is flagged
        if multi_patch:
            needs_global = bool(flag_of(truth[i])) or any(flag_of(truth[i + j]) for j in range(1, L + 1))
        else:
            needs_global = bool(flag_of(truth[i]))
        kk = len(draft)
        # patches run through the global model in this round: a0's, and (multi-patch) every flagged drafted atom,
        # speculatively, whether or not it is accepted
        n_patches = flag_of(truth[i]) + (sum(1 for dj in draft if flag_of(dj)) if multi_patch else 0)
        res.by_kp[(kk, n_patches)] = res.by_kp.get((kk, n_patches), 0) + 1
        res.rounds += 1
        res.by_k[kk] = res.by_k.get(kk, 0) + 1
        if needs_global:
            res.patch_rounds[kk] = res.patch_rounds.get(kk, 0) + 1
        res.global_passes += int(needs_global)
        res.drafted += len(draft)
        res.l_hist[L] = res.l_hist.get(L, 0) + 1
        emitted_atoms = truth[i + 1: i + 1 + take]
        if drafter is not None:
            drafter.push(emitted_atoms)
        res.emitted += take
        i += take
    return res
