"""Small token-LM drafters for crossing (multi-patch) drafts (Task B, Stage B0).

Protocol, per round (offline): the drafter greedily continues the text so far (prompt + accepted output) by M
tokens. The continuation is turned into bytes, cut to K bytes, and the boundary flags of the draft bytes come
from the dolma2 tokenization of context + draft. `build_draft` is the pure part (no model); `LMEngine` runs the
model; `PrecomputedLMDrafter` plugs recorded continuations into `sim.simulate`.

Two context modes: `raw` continues the text exactly as it is (a trailing incomplete UTF-8 character is held
back); `healed` also holds back the trailing word fragment and requires the continuation to start with it
(otherwise the raw continuation is used), like token healing.

Two flag modes: `cut` (the task's protocol: tokenize context + the draft cut to K bytes, drop the last,
possibly incomplete token and end the draft before it) and `full` (tokenize context + the whole M-token
continuation, drop only that text's last token, then cut to K bytes; flags are those of the longer text).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from . import make_atom
from .flags import Dolma2, spans_to_flags

_FRAGMENT = re.compile(r"\w+$")


def utf8_valid_prefix(data: bytes) -> int:
    try:
        data.decode("utf-8")
        return len(data)
    except UnicodeDecodeError as e:
        return e.start


def context_back(data: bytes, healed: bool) -> int:
    """Number of trailing bytes of `data` held back from the drafter's context."""
    valid = utf8_valid_prefix(data)
    back = len(data) - valid
    if healed:
        m = _FRAGMENT.search(data[:valid].decode("utf-8"))
        if m:
            back += len(m.group(0).encode("utf-8"))
    return back


@dataclass
class Cont:
    """A recorded continuation: the model continued data[:len(data) - back]; `cont` starts at that point."""

    back: int
    cont: bytes
    tok_ends: list[int]  # cumulative byte ends of the drafter's tokens in `cont`


def build_draft(ctx: bytes, c: Cont | None, k: int, dol: Dolma2, flag_mode: str = "cut") -> tuple[list[int], int]:
    """(draft atoms, drafter tokens needed to produce them). `ctx` = all bytes so far (a_0 included)."""
    if c is None or k <= 0:
        return [], 0
    back = ctx[len(ctx) - c.back:] if c.back else b""
    if not c.cont.startswith(back):
        return [], 0
    d = c.cont[len(back):]
    if not d:
        return [], 0
    n = len(ctx)
    if flag_mode == "cut":
        joint = ctx + d[:k]
        spans, covered = dol.token_spans(joint)
        end = spans[-2][1] if len(spans) >= 2 else 0  # drop the last, possibly incomplete token
        end = min(end, covered, n + k)
        flags = spans_to_flags(spans, len(joint))
    elif flag_mode == "full":
        joint = ctx + d
        spans, covered = dol.token_spans(joint)
        end = spans[-2][1] if len(spans) >= 2 else 0
        end = min(end, covered, n + k)
        flags = spans_to_flags(spans, len(joint))
    else:
        raise ValueError(flag_mode)
    if end <= n:
        return [], 0
    atoms = [make_atom(joint[p], flags[p]) for p in range(n, end)]
    need = len(back) + len(atoms)
    m_used = next((j + 1 for j, e in enumerate(c.tok_ends) if e >= need), len(c.tok_ends))
    return atoms, m_used


def build_token_draft(ctx: bytes, c: Cont | None, k_tok: int, dol: Dolma2) -> tuple[list[int], int]:
    """OLMo 2: dolma2 token ids of the draft text (tokenized on its own), without its last token, at most k_tok."""
    if c is None or k_tok <= 0:
        return [], 0
    back = ctx[len(ctx) - c.back:] if c.back else b""
    if not c.cont.startswith(back):
        return [], 0
    d = c.cont[len(back):]
    d = d[:utf8_valid_prefix(d)]
    if not d:
        return [], 0
    ids = dol.encode_ids(d.decode("utf-8"))
    ids = ids[:-1][:k_tok]
    if not ids:
        return [], 0
    nbytes = sum(len(dol.id_bytes(i)) for i in ids)
    need = len(back) + nbytes
    m_used = next((j + 1 for j, e in enumerate(c.tok_ends) if e >= need), len(c.tok_ends))
    return ids, m_used


class PrecomputedLMDrafter:
    """`sim.Drafter` over recorded continuations: item i is the continuation of the context a_0 .. a_i."""

    name = "lm"

    def __init__(self, items: Sequence[Cont | None], dol: Dolma2, flag_mode: str = "cut"):
        self.items, self.dol, self.flag_mode = items, dol, flag_mode
        self.ctx = bytearray()
        self.i = 0
        self.m_hist: dict[int, int] = {}

    def reset(self, ctx_atoms: Sequence[int], truth: Sequence[int] | None = None) -> None:
        self.ctx = bytearray(a & 255 for a in ctx_atoms)  # prompt bytes followed by a_0
        self.i = 0

    def propose(self, k: int) -> list[int]:
        atoms, m = build_draft(bytes(self.ctx), self.items[self.i], k, self.dol, self.flag_mode)
        if atoms:
            self.m_hist[m] = self.m_hist.get(m, 0) + 1
        return atoms

    def push(self, atoms: Sequence[int]) -> None:
        self.ctx.extend(a & 255 for a in atoms)
        self.i += len(atoms)


class PrecomputedTokenDrafter:
    """`tokens.simulate_tokens` drafter for OLMo 2 over recorded continuations (item i = after token i)."""

    name = "lm"

    def __init__(self, items: Sequence[Cont | None], dol: Dolma2):
        self.items, self.dol = items, dol
        self.ctx = bytearray()
        self.i = 0
        self.m_hist: dict[int, int] = {}

    def reset(self, ctx: Sequence[int], truth: Sequence[int] | None = None) -> None:
        # `ctx` = prompt token ids + t0; the byte text is supplied through `set_text`
        self.i = 0

    def set_text(self, prompt_bytes: bytes, first_token_bytes: bytes) -> None:
        self.ctx = bytearray(prompt_bytes + first_token_bytes)

    def propose(self, k: int) -> list[int]:
        ids, m = build_token_draft(bytes(self.ctx), self.items[self.i], k, self.dol)
        if ids:
            self.m_hist[m] = self.m_hist.get(m, 0) + 1
        return ids

    def push(self, toks: Sequence[int]) -> None:
        for t in toks:
            self.ctx.extend(self.dol.id_bytes(t))
        self.i += len(toks)


class LMEngine:
    """Greedy continuation with a small causal LM (HF transformers), batched with left padding."""

    def __init__(self, repo: str, revision: str, device: str = "cuda", dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.repo, self.revision, self.device = repo, revision, device
        self.tok = AutoTokenizer.from_pretrained(repo, revision=revision)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(repo, revision=revision, torch_dtype=getattr(torch, dtype)).to(device).eval()
        self.eos = {self.tok.eos_token_id} if self.tok.eos_token_id is not None else set()
        from .flags import _unicode_to_byte

        u2b = _unicode_to_byte()
        self._tokbytes: dict[int, bytes] = {}
        for t, i in self.tok.get_vocab().items():
            if all(ch in u2b for ch in t):
                self._tokbytes[i] = bytes(u2b[ch] for ch in t)
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def continue_texts(self, texts: Sequence[str], m: int, batch: int = 48) -> list[tuple[bytes, list[int]]]:
        """Greedy continuation by up to `m` tokens: (bytes, cumulative token byte ends) per text."""
        torch = self.torch
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out: list[tuple[bytes, list[int]] | None] = [None] * len(texts)
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
            with torch.inference_mode():
                gen = self.model.generate(**enc, max_new_tokens=m, do_sample=False, pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:].tolist()
            for i, ids in zip(idx, new):
                buf = b""
                ends: list[int] = []
                for t in ids:
                    if t in self.eos or t not in self._tokbytes:
                        break
                    buf += self._tokbytes[t]
                    ends.append(len(buf))
                valid = utf8_valid_prefix(buf)  # drop a trailing incomplete character
                while ends and ends[-1] > valid:
                    ends.pop()
                out[i] = (buf[:valid], ends)
        return out  # type: ignore[return-value]


def record_continuations(engine: LMEngine, ctx_bytes: Sequence[bytes], healed: bool, m: int,
                         memo: dict[str, tuple[bytes, list[int]]], back_fn=None) -> list[Cont | None]:
    """One continuation per context (each a byte string ending with the last emitted byte).
    `back_fn(data)` (optional) replaces `context_back` to choose how many trailing bytes are held back."""
    plan: list[tuple[int, str] | None] = []
    for data in ctx_bytes:
        back = back_fn(data) if back_fn is not None else context_back(data, healed)
        text = data[:len(data) - back].decode("utf-8")
        plan.append((back, text) if text else None)
    need = sorted({t for p in plan if p for t in [p[1]] if t not in memo})
    if need:
        for t, r in zip(need, engine.continue_texts(need, m)):
            memo[t] = r
    res: list[Cont | None] = []
    for p in plan:
        if p is None:
            res.append(None)
            continue
        back, text = p
        cont, ends = memo[text]
        # the continuation of text = data[:len-back]; build_draft checks that it starts with the held-back bytes
        res.append(Cont(back, cont, ends))
    return res


def merge_healed(raw: Sequence[Cont | None], healed: Sequence[Cont | None], ctxs: Sequence[bytes]) -> list[Cont | None]:
    """Healed continuation where it starts with the held-back fragment, otherwise the raw one."""
    out: list[Cont | None] = []
    for r, h, data in zip(raw, healed, ctxs):
        if h is not None and h.cont.startswith(data[len(data) - h.back:] if h.back else b""):
            out.append(h)
        else:
            out.append(r)
    return out
