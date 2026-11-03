"""Drafter diagnostics, analysis 6: per-byte cross-entropy of each model's output under each drafter.

Teacher-forced, bf16, one forward per text (prompt + output; the loss covers the tokens that start at or after the
prompt end, in the drafter's own tokenization; bits per byte = summed NLL / bytes of those tokens).
Texts: Bolmo's recorded output (256 new bytes) and OLMo 2's greedy output (t0 + 71 tokens), extended prompt set.

Usage: python tools/diag_ce.py --name smollm2-360m
Output: results/diagnostics/ce_<name>.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch  # noqa: E402

from byteprof.drafting import byte_of  # noqa: E402
from byteprof.drafting.diagnostics import bits_per_byte  # noqa: E402
from byteprof.drafting.flags import Dolma2  # noqa: E402
from byteprof.drafting.lmdraft import LMEngine, utf8_valid_prefix  # noqa: E402
from byteprof.drafting.truthio import load_truth  # noqa: E402
from cross_record_drafts import DRAFTERS  # noqa: E402


def text_nll(eng: LMEngine, prompt: bytes, out: bytes) -> tuple[list[float], list[int]]:
    """Per-token NLL (nats) and byte length of the tokens that start at or after the end of `prompt`."""
    joined = prompt + out  # t0 of OLMo can end inside a character, so only the joined bytes are valid UTF-8
    text = joined[:utf8_valid_prefix(joined)].decode("utf-8")
    enc = eng.tok(text, return_tensors="pt").to(eng.device)
    ids = enc["input_ids"][0]
    with torch.inference_mode():
        logits = eng.model(**enc).logits[0].float()
    logp = torch.log_softmax(logits[:-1], dim=-1)
    nll = (-logp.gather(1, ids[1:, None])[:, 0]).tolist()
    pos = 0
    nlls, nbytes = [], []
    for j, t in enumerate(ids.tolist()):
        n = len(eng._tokbytes.get(t, b""))
        if j >= 1 and pos >= len(prompt) and n:
            nlls.append(nll[j - 1])
            nbytes.append(n)
        pos += n
    return nlls, nbytes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=sorted(DRAFTERS))
    a = ap.parse_args()
    repo, rev = DRAFTERS[a.name]
    dol = Dolma2()
    eng = LMEngine(repo, rev)
    cases = load_truth(ROOT / "results/drafting/truth/bf16.json")
    olmo = json.loads((ROOT / "results/drafting/truth/olmo-bf16.json").read_text(encoding="utf-8"))["prompts"]
    res: dict = {"name": a.name, "bolmo": {}, "olmo": {}}
    tot = {"bolmo": ([], []), "olmo": ([], [])}
    by_type: dict = {}
    for pid, c in cases.items():
        if c.set == "default":
            continue
        prompt = c.text.encode("utf-8")
        out_b = bytes(byte_of(x) for x in c.truth[256][1:])
        o = olmo[pid]
        toks = o["tokens"]["72"]
        prompt2 = prompt + dol.id_bytes(toks[0])
        out_o = b"".join(dol.id_bytes(t) for t in toks[1:])
        for text, (pr, out) in (("bolmo", (prompt, out_b)), ("olmo", (prompt2, out_o))):
            nl, nb = text_nll(eng, pr, out)
            res[text][pid] = {"bpb": bits_per_byte(nl, nb), "bytes": sum(nb)}
            tot[text][0].extend(nl)
            tot[text][1].extend(nb)
            k = (text, c.type)
            by_type.setdefault(k, ([], []))
            by_type[k][0].extend(nl)
            by_type[k][1].extend(nb)
    for text in ("bolmo", "olmo"):
        res[text]["all"] = {"bpb": bits_per_byte(*tot[text]), "bytes": sum(tot[text][1])}
    res["by_type"] = {f"{t}|{ty}": bits_per_byte(*v) for (t, ty), v in by_type.items()}
    (ROOT / "results/diagnostics" / f"ce_{a.name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(a.name, {t: round(res[t]["all"]["bpb"], 3) for t in ("bolmo", "olmo")}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
