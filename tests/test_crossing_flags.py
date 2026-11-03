"""Task B offline pieces: flag agreement with a tokenization, and the crossing-draft protocol (no model, no download).

The protocol tests use a fake tokenizer (a token = optional spaces + a run of non-space bytes) so they run anywhere;
two tests use the real dolma2 tokenizer when it is cached."""

import re

import pytest

from byteprof.drafting import flag_of, make_atom
from byteprof.drafting.flags import HF_CACHE, Dolma2, classify_diff, compare_flags, spans_to_flags
from byteprof.drafting.lmdraft import (Cont, PrecomputedLMDrafter, build_draft, build_token_draft, context_back,
                                       merge_healed)
from byteprof.drafting.sim import simulate

TOK = re.compile(rb"\s*\S+")


class FakeDol:
    def token_spans(self, data: bytes):
        try:
            used = len(data.decode("utf-8").encode("utf-8"))
        except UnicodeDecodeError as e:
            used = e.start
        return [(m.start(), m.end()) for m in TOK.finditer(data[:used])], used

    def encode_ids(self, text: str):
        return [hash(m.group(0)) % 100000 for m in TOK.finditer(text.encode())]

    def id_bytes(self, i):  # unused by the fake ids
        return b""


def flags_of(text: bytes, dol=FakeDol()):
    spans, _ = dol.token_spans(text)
    return spans_to_flags(spans, len(text))


def test_compare_flags_perfect_and_flipped():
    text = b"the cat sat on the mat and then it left"
    dol = FakeDol()
    spans, covered = dol.token_spans(text)
    f = spans_to_flags(spans, len(text))
    r = compare_flags(f, spans, 0, covered)
    # the last token is excluded, everything before agrees
    assert r["agree"] == r["positions"] == spans[-2][1]
    assert r["patches_exact"] == r["patches"] == len(spans) - 1
    g = list(f)
    p = spans[2][1] - 1  # drop one boundary: two tokens merge
    g[p] = 0
    r = compare_flags(g, spans, 0, covered)
    assert r["positions"] - r["agree"] == 1 and r["diffs"] == [(p, 0, 1)]
    assert r["patches"] == len(spans) - 2 and r["patches_exact"] == r["patches"] - 1


def test_compare_flags_ignores_final_token_and_prompt_region():
    text = b"aa bb cc dd"
    spans, covered = FakeDol().token_spans(text)
    f = spans_to_flags(spans, len(text))
    f[-1] = 0  # the final token is not evaluated
    f[2] = 0   # a prompt-region disagreement is outside `first`
    r = compare_flags(f, spans, 3, covered)
    assert r["agree"] == r["positions"]
    assert r["positions"] == spans[-2][1] - 3


def test_classify_diff_kinds():
    assert classify_diff("straße".encode(), 4, 0).endswith("multibyte")
    assert classify_diff(b"abc 123", 4, 1) == "extra/whitespace" or classify_diff(b"abc 123", 4, 1).endswith("digit")
    assert classify_diff(b"abcdef", 2, 0) == "missing/inside-word"
    assert classify_diff(b"abc.def", 3, 1).startswith("extra/")


def test_context_back():
    assert context_back(b"hello wor", False) == 0
    assert context_back(b"hello wor", True) == 3
    assert context_back(b"hello ", True) == 0
    assert context_back("grü".encode()[:-1], False) == 1  # incomplete character is always held back
    assert context_back("grü".encode()[:-1], True) == 3  # 1 held-back byte + the fragment "gr"


def cont(back: int, text: bytes) -> Cont:
    ends, pos = [], 0
    for m in TOK.finditer(text):
        pos = m.end()
        ends.append(pos)
    return Cont(back, text, ends)


def test_build_draft_cut_drops_last_token_and_flags_follow_the_retokenization():
    dol = FakeDol()
    ctx = b"the cat sat"
    c = cont(0, b" on the mat and then")
    atoms, m = build_draft(ctx, c, 12, dol, "cut")
    # draft cut to 12 bytes = " on the mat " ; tokens " on", " the", " mat" -> last one dropped
    draft = bytes(a & 255 for a in atoms)
    assert draft == b" on the"
    assert [flag_of(a) for a in atoms] == flags_of(ctx + draft)[len(ctx):]
    assert flag_of(atoms[-1]) == 1  # the draft ends on a token end
    assert m == 2
    full, _ = build_draft(ctx, c, 12, dol, "full")
    assert bytes(a & 255 for a in full) == b" on the mat "[:12]
    assert len(full) >= len(atoms)


def test_build_draft_joins_a_word_across_the_seam():
    dol = FakeDol()
    ctx = b"the Eisenb"
    c = cont(0, b"ahn fuhr los")
    atoms, _ = build_draft(ctx, c, 32, dol, "cut")
    # the first token of the draft continues "Eisenb", so no flag inside it
    assert bytes(a & 255 for a in atoms) == b"ahn fuhr"
    assert flags_of(ctx + b"ahn fuhr")[len(ctx):] == [flag_of(a) for a in atoms]
    assert [flag_of(a) for a in atoms][:3] == [0, 0, 1]


def test_build_draft_healed_requires_the_fragment_and_cut_never_ends_in_the_middle_of_a_character():
    dol = FakeDol()
    ctx = b"die Eisenb"
    assert build_draft(ctx, Cont(6, b"Eisenbahn fuhr los", [9, 13, 18]), 32, dol, "cut")[0] != []
    assert build_draft(ctx, Cont(6, b"Strasse fuhr los", [7, 11, 16]), 32, dol, "cut")[0] == []
    atoms, _ = build_draft(b"x ", cont(0, "grüße und mehr".encode()), 4, dol, "cut")
    bytes(a & 255 for a in atoms).decode("utf-8")  # valid UTF-8
    assert build_draft(b"x", None, 4, dol)[0] == [] and build_draft(b"x", cont(0, b" a b"), 0, dol)[0] == []


def test_merge_healed_falls_back_to_raw():
    ctx = b"die Eisenb"
    raw = [cont(0, b"ahn fuhr")]
    ok = [Cont(6, b"Eisenbahn fuhr", [9, 14])]
    bad = [Cont(6, b"Strasse fuhr", [7, 12])]
    assert merge_healed(raw, ok, [ctx])[0] is ok[0]
    assert merge_healed(raw, bad, [ctx])[0] is raw[0]


def _truth(text: bytes):
    f = flags_of(text)
    return [make_atom(b, fl) for b, fl in zip(text, f)]


def test_replay_in_the_simulator_matches_a_hand_computation():
    """A recorded perfect continuation (the true next bytes) accepts across patches; an empty one never does."""
    dol = FakeDol()
    prompt = b"one two "
    out = b"three four five six seven eight nine ten eleven twelve"
    full = prompt + out
    atoms = _truth(full)
    ctx_atoms = atoms[:len(prompt) - 1]
    truth = atoms[len(prompt) - 1:]  # a_0 = last prompt byte
    items = [cont(0, full[len(prompt) + i:]) for i in range(len(truth))]  # continuation of everything up to a_i
    # item i continues data[:len(data)] where data = prompt[:-1] + a_0 .. a_i, i.e. the next bytes of `full`
    d = PrecomputedLMDrafter(items, dol, "full")
    res = simulate(truth, d, 16, ctx_atoms=ctx_atoms, max_new=len(truth) - 1, multi_patch=True)
    assert res.rounds < len(truth) - 1  # drafts were accepted across boundaries
    assert res.emitted == len(truth) - 1
    assert res.bytes_per_local_pass > 4
    none = PrecomputedLMDrafter([None] * len(truth), dol, "full")
    r0 = simulate(truth, none, 16, ctx_atoms=ctx_atoms, max_new=len(truth) - 1, multi_patch=True)
    assert r0.rounds == len(truth) - 1 and r0.bytes_per_local_pass == 1.0


def test_wrong_flags_end_acceptance_in_the_multi_patch_rule():
    """Right bytes with a flag Bolmo does not predict are rejected (byte and flag must both match)."""
    dol = FakeDol()
    truth = _truth(b"ab cd ef gh")
    ctx = truth[:1]
    t = truth[1:]
    wrong = [(a & 255) | ((1 - flag_of(a)) << 8) for a in t]

    class Fixed:
        def reset(self, *_):
            pass

        def propose(self, k):
            return wrong[:k]

        def push(self, atoms):
            pass

    r = simulate(truth[:], Fixed(), 4, ctx_atoms=[], multi_patch=True, max_new=len(truth) - 1)
    assert r.l_hist.get(0, 0) == r.rounds  # nothing is ever accepted


@pytest.mark.skipif(not list((HF_CACHE / "models--allenai--OLMo-2-0425-1B" / "snapshots").glob("*/tokenizer.json")),
                    reason="OLMo 2 tokenizer not cached")
def test_dolma2_spans_cover_the_bytes_and_match_the_flags_of_the_corpus_convention():
    dol = Dolma2()
    text = "Die Straße nach Zürich: 12345 中文 \U0001f600 end.\n    return x"
    data = text.encode()
    spans, covered = dol.token_spans(data)
    assert covered == len(data) and spans[0][0] == 0 and spans[-1][1] == len(data)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert sum(len(dol.id_bytes(i)) for i in dol.encode_ids(text)) == len(data)
    # incomplete trailing character: tokenized prefix only
    spans2, covered2 = dol.token_spans(data[:-0 or None][: data.index("ü".encode()) + 1])
    assert covered2 == data.index("ü".encode())
    assert build_token_draft(b"x", None, 4, dol) == ([], 0)
