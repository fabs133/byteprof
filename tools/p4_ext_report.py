"""Stage 2 step 2 (TASK_build_p4.md): acceptance of the real gated loop on the extended prompt set against the simulation
(results/p4/tallies_<prec>.json, same K). -> results/p4/ext_report.{md,json}"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELLS = {"bf16": ("bf16-p4-ext", "ext|P4|32|6"), "int4": ("int4-p4-ext", "ext|P4|16|4")}


def main() -> int:
    md = ["# P4 acceptance on the extended prompt set (25 prompts, 256 new bytes, one pass), real loop vs simulation", "",
          "| precision | | rounds | B/round | B/global pass | boundary rounds (crossing attempted) | of which multi-patch | bytes in multi-patch rounds | patches per boundary round | cap cuts | ms/byte |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    out: dict = {}
    for prec, (cell, key) in CELLS.items():
        d = json.loads((ROOT / f"results/p4/speed/ext/{cell}.json").read_text(encoding="utf-8"))
        rows = d["rows"]
        nb = sum(r["new_bytes"] for r in rows)
        rounds = sum(r["rounds"] for r in rows)
        glob = sum(r["global_calls"] for r in rows)
        cross = sum(r["kinds"]["cross"] for r in rows)
        cross_b = sum(r["kind_emitted"]["cross"] for r in rows)
        pk = sum(r["patches_kept"] for r in rows)
        gp = sum(r["global_patches"] for r in rows)
        ph: dict = {}
        for r in rows:
            for k, v in r["patches_hist"].items():
                ph[int(k)] = ph.get(int(k), 0) + v
        patches = (sum(k * v for k, v in ph.items()) + (glob - cross)) / max(glob, 1)  # boundary rounds without a flagged draft carry one patch
        cuts = sum(r["cap_cut"] for r in rows)
        real = {"rounds": rounds, "bpr": nb / rounds, "bpg": nb / glob, "cross_share": glob / rounds, "multi_share": cross / rounds, "cross_bytes": cross_b / nb,
                "patches": patches, "cap_cuts": cuts, "ms_per_byte": d["ms_per_byte"], "emitted": nb, "patches_kept": pk,
                "global_patches": gp, "hist": ph}
        t = json.loads((ROOT / f"results/p4/tallies_{prec}.json").read_text(encoding="utf-8"))["cells"][key]
        sim_cross_patches = [(int(k.split(",")[1]), v) for k, v in t["by_kp"].items() if int(k.split(",")[2])]
        # simulated patches of crossing rounds that run a multi-patch pass (P >= 2) and all crossing rounds
        sim_p = sum(p * v for p, v in sim_cross_patches) / max(sum(v for _, v in sim_cross_patches), 1)
        sim = {"rounds": t["rounds"], "bpr": t["emitted"] / t["rounds"], "bpg": t["emitted"] / t["global"],
               "cross_share": t["kinds"]["cross"] / t["rounds"], "patches": sim_p, "emitted": t["emitted"]}
        out[prec] = {"real": real, "sim": sim, "key": key}
        md.append(f"| {prec} | real loop | {rounds} | {real['bpr']:.2f} | {real['bpg']:.2f} | {real['cross_share'] * 100:.0f}% | {real['multi_share'] * 100:.0f}% | {real['cross_bytes'] * 100:.0f}% | "
                  f"{patches:.2f} | {cuts} | {d['ms_per_byte']:.3f} |")
        md.append(f"| {prec} | simulation ({key}) | {sim['rounds']} | {sim['bpr']:.2f} | {sim['bpg']:.2f} | {sim['cross_share'] * 100:.0f}% | - | - | {sim_p:.2f} | - | - |")
    md += ["", "(The simulation replays the recorded minimal-loop text, the real loop produces its own text where it diverges at near-ties. "
           "Boundary rounds = rounds that start at a patch boundary: the crossing rule is tried there; `multi-patch` = those whose draft carried at least "
           "one flag. Patches per boundary round counts a boundary round without a flagged draft as one patch, as the simulation does.)", ""]
    for prec in CELLS:
        ph = out[prec]["real"]["hist"]
        md.append(f"{prec}: patches per multi-patch round (real): " + ", ".join(f"{k}: {v}" for k, v in sorted(ph.items())))
    (ROOT / "results/p4/ext_report.md").write_text("\n".join(md), encoding="utf-8")
    (ROOT / "results/p4/ext_report.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
