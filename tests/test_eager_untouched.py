"""The default (optimize = "none") path must never import byteprof.optim."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_default_config_does_not_import_optim():
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
        "from byteprof.config import load_config\n"
        "from byteprof import runner, cli\n"
        "cfg = load_config(None)\n"
        "assert cfg.optimize == 'none' and cfg.decode_loop == 'hf' and cfg.instrument == 'none'\n"
        "assert not [m for m in sys.modules if m.startswith('byteprof.optim')]\n"
        "bad = lambda: [m for m in sys.modules if m.startswith('byteprof.quant') or m.split('.')[0] == 'torchao']\n"
        "assert not bad()\n"
        "cfg = load_config(None, {'quantize': {'target': 'none', 'scheme': 'int4_wo'}})\n"
        "assert cfg.quantize['target'] == 'none' and not bad()\n"
    )
    r = subprocess.run([sys.executable, "-I", "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_profile_dispatches_to_optimized_only_when_requested():
    src = (ROOT / "byteprof" / "runner.py").read_text(encoding="utf-8")
    assert src.count("from . import optim") == 1
    assert src.index("from . import optim") > src.index("def profile_optimized")


def test_quantization_code_imported_only_behind_the_target_guard():
    src = (ROOT / "byteprof" / "runner.py").read_text(encoding="utf-8")
    assert src.count("from .quant") == 1
    guard = src.index('cfg.quantize.get("target", "none") != "none"')
    assert guard < src.index("from .quant") < guard + 200
    cfg_src = (ROOT / "byteprof" / "config.py").read_text(encoding="utf-8")
    assert not [l for l in cfg_src.splitlines() if l.startswith(("import ", "from ")) and "quant" in l]


# ---- drafting (TASK_drafting.md): `draft.kind = "none"` leaves the minimal loop and the default path unchanged ----

BOLMO_LOOP_SHA256 = "a5b090ba7ee5074087b9d85c6ed8f9ee1f9238e3380faffb5487626472361014"  # as of the quantization task


def test_draft_none_or_missing_never_imports_the_drafting_code():
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
        "from byteprof.config import load_config\n"
        "from byteprof import runner, cli\n"
        "for ov in (None, {'draft': {}}, {'draft': {'kind': 'none'}}):\n"
        "    cfg = load_config(None, ov)\n"
        "    assert cfg.draft.get('kind', 'none') == 'none'\n"
        "bad = [m for m in sys.modules if m.startswith('byteprof.drafting') or m.startswith('byteprof.optim')]\n"
        "assert not bad, bad\n"
    )
    r = subprocess.run([sys.executable, "-I", "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_the_minimal_loop_file_is_unchanged_and_does_not_know_about_drafting():
    import hashlib

    text = (ROOT / "byteprof" / "optim" / "bolmo_loop.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "draft" not in text.lower()
    assert hashlib.sha256(text.encode()).hexdigest() == BOLMO_LOOP_SHA256


def test_draft_code_is_imported_only_behind_the_kind_guard():
    src = (ROOT / "byteprof" / "optim" / "bolmo.py").read_text(encoding="utf-8")
    assert src.count("from . import bolmo_draft") == 1
    guard = src.index('if cfg.draft.get("kind", "none") != "none"')
    assert guard < src.index("from . import bolmo_draft") < guard + 200
    for f in ("runner.py", "config.py", "cli.py"):
        text = (ROOT / "byteprof" / f).read_text(encoding="utf-8")
        imports = "\n".join(l for l in text.splitlines() if l.lstrip().startswith(("import ", "from ")))
        assert not any(w in imports for w in ("drafting", "bolmo_draft", "draft_core")), f


# ---- fused verification (TASK_parallel_verify.md): `draft.verify = "steps"` (the default) is today's path ----


def test_fused_verification_code_is_imported_only_behind_the_verify_guard():
    src = (ROOT / "byteprof" / "optim" / "bolmo_draft.py").read_text(encoding="utf-8")
    assert src.count("from . import draft_fused") == 2  # the layer and the commit, both inside `verify == "fused"`
    for m in __import__("re").finditer("from . import draft_fused", src):
        assert 'self.verify == "fused"' in src[max(0, m.start() - 200):m.start()]
    for f in ("bolmo.py", "olmo.py", "olmo_draft.py", "draft_core.py", "bolmo_loop.py"):
        text = (ROOT / "byteprof" / "optim" / f).read_text(encoding="utf-8")
        assert "draft_fused" not in text and "mlstm_multistep" not in text, f


def test_the_default_verify_path_never_imports_the_fused_kernels():
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
        "from byteprof.config import load_config\n"
        "import byteprof.optim.bolmo_draft, byteprof.optim.draft_core\n"
        "cfg = load_config(None, {'optimize': 'cuda_graph', 'decode_loop': 'minimal', 'optimize_global': True,\n"
        "                         'draft': {'kind': 'combined'}})\n"
        "assert cfg.draft.get('verify', 'steps') == 'steps'\n"
        "bad = [m for m in sys.modules if m.endswith(('draft_fused', 'mlstm_multistep')) or m.startswith('triton')]\n"
        "assert not bad, bad\n"
    )
    r = subprocess.run([sys.executable, "-I", "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ---- boundary-gated drafting (TASK_build_p4.md): `draft.policy = "within"` (the default) is today's path ----

DRAFT_CORE_SHA256 = "f72d5fd9fdd724dfd376e47b7fecb7bf174c244e57b37635a826e7746b1f1359"  # as of the parallel-verify task


def _lf(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_gated_code_is_imported_only_behind_the_policy_guard():
    import re

    src = _lf(ROOT / "byteprof" / "optim" / "bolmo_draft.py")
    guard = 'spec.get("policy", "within") == "gated"'
    for name in ("from . import draft_gated", "from . import bolmo_gated", "from ..drafting.flags import Dolma2"):
        found = list(re.finditer(re.escape(name), src))
        assert found, name
        for m in found:
            assert guard in src[max(0, m.start() - 120):m.start()], name
    for f in ("bolmo.py", "olmo.py", "olmo_draft.py", "draft_core.py", "bolmo_loop.py", "draft_fused.py"):
        text = _lf(ROOT / "byteprof" / "optim" / f)
        assert "draft_gated" not in text and "bolmo_gated" not in text, f
    for f in ("runner.py", "config.py", "cli.py"):
        text = _lf(ROOT / "byteprof" / f)
        assert "draft_gated" not in text and "bolmo_gated" not in text, f


def test_the_default_policy_never_imports_the_gated_code():
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
        "from byteprof.config import load_config\n"
        "import byteprof.optim.bolmo_draft, byteprof.optim.draft_core\n"
        "cfg = load_config(None, {'optimize': 'cuda_graph', 'decode_loop': 'minimal', 'optimize_global': True,\n"
        "                         'draft': {'kind': 'combined', 'policy': 'within'}})\n"
        "assert cfg.draft.get('policy', 'within') == 'within'\n"
        "bad = [m for m in sys.modules if m.endswith(('draft_gated', 'bolmo_gated', 'drafting.flags')) or m == 'tokenizers']\n"
        "assert not bad, bad\n"
    )
    r = subprocess.run([sys.executable, "-I", "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_the_core_loop_is_unchanged_and_the_default_engine_adds_no_graphs():
    import hashlib

    text = _lf(ROOT / "byteprof" / "optim" / "draft_core.py")
    assert "gated" not in text.lower()
    assert hashlib.sha256(text.encode()).hexdigest() == DRAFT_CORE_SHA256
    src = _lf(ROOT / "byteprof" / "optim" / "bolmo_draft.py")
    hook = src[src.index("def _extra_fns"):]
    assert hook.split("def capture")[0].strip().endswith("return []")  # the base engine adds no crossing graphs
