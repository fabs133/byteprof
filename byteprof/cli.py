"""Command line entry point: `python -m byteprof {discover,inspect,profile}`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import ConfigError, load_config, validate

EXIT_OK, EXIT_ERROR, EXIT_CONFIG, EXIT_MODEL = 0, 1, 2, 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="byteprof", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", type=Path, help="TOML config file")
        sp.add_argument("--model-id")
        sp.add_argument("--revision")
        sp.add_argument("--device")
        sp.add_argument("--dtype")
        sp.add_argument("--output-dir", type=Path)

    d = sub.add_parser("discover", help="dump the module tree and suggest component patterns")
    common(d)
    d.add_argument("--max-depth", type=int, default=4)

    ins = sub.add_parser("inspect", help="one generation per prompt; check byte counting (exit 2 if a row fails)")
    common(ins)
    ins.add_argument("--prompts-file", type=Path)
    ins.add_argument("--max-new-bytes", type=int)
    ins.add_argument("--seed", type=int)

    pr = sub.add_parser("profile", help="run the per-component latency profile")
    common(pr)
    pr.add_argument("--prompts-file", type=Path)
    pr.add_argument("--max-new-bytes", type=int)
    pr.add_argument("--runs", type=int, dest="measured_runs")
    pr.add_argument("--warmup", type=int, dest="warmup_runs")
    pr.add_argument("--seed", type=int)
    pr.add_argument("--optimize", choices=["none", "compile", "cuda_graph"])
    pr.add_argument("--instrument", choices=["none", "timing", "syncs"])
    pr.add_argument("--decode-loop", choices=["hf", "minimal"], dest="decode_loop")
    pr.add_argument("--trace", action="store_true", default=None, help="also export a torch.profiler trace")

    ev = sub.add_parser("eval", help="quality of the (quantized) model: bits per byte or CUTE accuracy")
    common(ev)
    ev.add_argument("--task", choices=["bpb", "cute"], required=True)
    ev.add_argument("--limit", type=int, help="bpb: number of 1024-byte chunks; cute: number of items")
    ev.add_argument("--seed", type=int)
    return p


_OVERRIDE_KEYS = (
    "model_id", "revision", "device", "dtype", "output_dir", "prompts_file",
    "max_new_bytes", "measured_runs", "warmup_runs", "seed", "trace", "optimize", "instrument", "decode_loop",
)


def _table(rows: list[dict]) -> str:
    cols = ["prompt_id", "prompt_bytes", "input_len", "output_len", "new_bytes", "step_calls", "max_new", "ok"]
    body = [[str(r[c]) for c in cols] for r in rows]
    widths = [max(len(c), *(len(b[i]) for b in body)) for i, c in enumerate(cols)]
    fmt = lambda cells: "  ".join(c.rjust(w) for c, w in zip(cells, widths))  # noqa: E731
    return "\n".join([fmt(cols)] + [fmt(b) for b in body])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("byteprof")
    overrides = {k: getattr(args, k, None) for k in _OVERRIDE_KEYS}

    try:
        cfg = load_config(args.config, overrides)
        if args.command == "profile" and cfg.optimize == "none":  # optimized runs have no breakdown
            validate(cfg, require_components=True)
    except ConfigError as exc:
        log.error("config error: %s", exc)
        return EXIT_CONFIG

    # Imported lazily so `--help` and config errors work without torch installed.
    from .runner import ModelLoadError, discover, inspect, profile

    try:
        if args.command == "discover":
            path = discover(cfg, max_depth=args.max_depth)
            log.info("module tree written to %s", path)
        elif args.command == "inspect":
            path, rows = inspect(cfg)
            print(_table(rows))
            log.info("inspect results written to %s", path)
            if not all(r["ok"] for r in rows):
                log.error("inspect: %d row(s) not ok", sum(not r["ok"] for r in rows))
                return EXIT_CONFIG
        elif args.command == "eval":
            from .evaluate import evaluate

            log.info("eval written to %s", evaluate(cfg, args.task, args.limit, args.output_dir))
        else:
            out = profile(cfg)
            log.info("results written to %s", out)
    except ModelLoadError as exc:
        log.error("model load failed: %s", exc)
        return EXIT_MODEL
    except (ValueError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return EXIT_CONFIG
    except Exception:
        log.exception("unexpected failure")
        return EXIT_ERROR
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
