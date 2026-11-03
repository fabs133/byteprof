from pathlib import Path

import pytest

from byteprof.config import ConfigError, load_config


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "c.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_precedence_defaults_file_cli(tmp_path):
    path = _write(tmp_path, 'max_new_bytes = 64\nseed = 7\n[components]\ng = ["global"]\n')
    cfg = load_config(path, {"max_new_bytes": 32, "seed": None}, require_components=True)
    assert cfg.max_new_bytes == 32  # CLI wins
    assert cfg.seed == 7  # file wins over default; None override ignored
    assert cfg.measured_runs == 5  # default
    assert cfg.components == {"g": ["global"]}


def test_unknown_key_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(_write(tmp_path, "max_new_bites = 3\n"))


def test_profile_requires_components():
    with pytest.raises(ConfigError, match="discover"):
        load_config(None, require_components=True)


def test_invalid_values_rejected():
    with pytest.raises(ConfigError):
        load_config(None, {"measured_runs": 0})
    with pytest.raises(ConfigError):
        load_config(None, {"dtype": "int4"})


def test_optimize_defaults_to_none_and_is_validated(tmp_path):
    assert load_config(None).optimize == "none"
    assert load_config(None, {"optimize": "cuda_graph"}).optimize == "cuda_graph"
    cfg = load_config(_write(tmp_path, 'optimize = "compile"\ncompile_backend = "cudagraphs"\n'))
    assert (cfg.optimize, cfg.compile_backend) == ("compile", "cudagraphs")
    with pytest.raises(ConfigError, match="optimize"):
        load_config(None, {"optimize": "graphs"})
    with pytest.raises(ConfigError, match="compile_backend"):
        load_config(None, {"compile_backend": "tensorrt"})


def test_decode_loop_and_instrument_validated_and_need_optimize():
    cfg = load_config(None)
    assert (cfg.decode_loop, cfg.instrument, cfg.optimize_global) == ("hf", "none", False)
    ok = load_config(None, {"optimize": "cuda_graph", "decode_loop": "minimal", "instrument": "timing"})
    assert (ok.decode_loop, ok.instrument) == ("minimal", "timing")
    with pytest.raises(ConfigError, match="decode_loop"):
        load_config(None, {"optimize": "compile", "decode_loop": "fast"})
    with pytest.raises(ConfigError, match="instrument"):
        load_config(None, {"optimize": "compile", "instrument": "all"})
    with pytest.raises(ConfigError, match="optimize != 'none'"):
        load_config(None, {"decode_loop": "minimal"})
    with pytest.raises(ConfigError, match="optimize != 'none'"):
        load_config(None, {"instrument": "syncs"})


def test_quantize_table_defaults_and_validation(tmp_path):
    assert load_config(None).quantize == {}
    cfg = load_config(_write(tmp_path, '[quantize]\ntarget = "local"\nscheme = "int4_wo"\ngroup_size = 128\n'))
    assert cfg.quantize == {"target": "local", "scheme": "int4_wo", "group_size": 128}
    for bad in ('target = "everything"', 'target = "local"\nscheme = "fp4"',
                'target = "local"\nscheme = "int4_wo"\ngroup_size = 100', 'target = "local"\nbogus = 1'):
        with pytest.raises(ConfigError):
            load_config(_write(tmp_path, f"[quantize]\n{bad}\n"))


def test_quantize_constants_match_selection_module():
    from byteprof import config
    from byteprof.quant import select

    assert config.VALID_QUANT_TARGETS == select.VALID_TARGETS
    assert config.VALID_QUANT_SCHEMES == select.VALID_SCHEMES


def test_draft_table_is_validated(tmp_path):
    cfg = load_config(None)
    assert cfg.draft == {}
    base = {"optimize": "cuda_graph", "optimize_global": True, "decode_loop": "minimal"}
    ok = load_config(None, {**base, "draft": {"kind": "combined", "ks": [0, 1, 4], "k_max": 4}})
    assert ok.draft["kind"] == "combined"
    assert load_config(None, {"draft": {"kind": "none"}}).draft["kind"] == "none"  # none needs nothing else
    with pytest.raises(ConfigError, match="draft.kind"):
        load_config(None, {**base, "draft": {"kind": "magic"}})
    with pytest.raises(ConfigError, match="draft must be a table"):
        load_config(None, {**base, "draft": {"kind": "context", "kmax": 4}})
    with pytest.raises(ConfigError, match="decode_loop"):
        load_config(None, {"optimize": "cuda_graph", "optimize_global": True, "draft": {"kind": "context"}})
    with pytest.raises(ConfigError, match="draft.ks"):
        load_config(None, {**base, "draft": {"kind": "context", "ks": [0, 99]}})
    with pytest.raises(ConfigError, match="truth_file"):
        load_config(None, {**base, "draft": {"kind": "oracle"}})
    path = _write(tmp_path, 'optimize = "cuda_graph"\noptimize_global = true\ndecode_loop = "minimal"\n'
                  '[draft]\nkind = "context"\nk_max = 4\n')
    assert load_config(path).draft == {"kind": "context", "k_max": 4}


def test_draft_verify_and_k_range_are_validated():
    base = {"optimize": "cuda_graph", "optimize_global": True, "decode_loop": "minimal"}
    assert load_config(None, {**base, "draft": {"kind": "context"}}).draft.get("verify", "steps") == "steps"
    ok = load_config(None, {**base, "draft": {"kind": "combined", "ks": [0, 1, 32], "k_max": 32, "verify": "fused"}})
    assert ok.draft["verify"] == "fused"
    with pytest.raises(ConfigError, match="draft.verify"):
        load_config(None, {**base, "draft": {"kind": "combined", "verify": "magic"}})
    with pytest.raises(ConfigError, match="draft.ks"):
        load_config(None, {**base, "draft": {"kind": "combined", "ks": [0, 33], "k_max": 4}})
    with pytest.raises(ConfigError, match="draft.ks"):
        load_config(None, {**base, "draft": {"kind": "combined", "ks": [0, 4], "k_max": 33}})


def test_draft_policy_is_validated():
    base = {"optimize": "cuda_graph", "optimize_global": True, "decode_loop": "minimal"}
    assert load_config(None, {**base, "draft": {"kind": "combined"}}).draft.get("policy", "within") == "within"
    ok = load_config(None, {**base, "draft": {"kind": "combined", "policy": "gated", "k_cross": 32, "max_patches": 8,
                                              "ks": [0, 1, 32], "k_max": 6, "verify": "fused"}})
    assert ok.draft["policy"] == "gated"
    with pytest.raises(ConfigError, match="draft.policy"):
        load_config(None, {**base, "draft": {"kind": "combined", "policy": "magic"}})
    with pytest.raises(ConfigError, match="lookup drafter"):
        load_config(None, {**base, "draft": {"kind": "oracle", "policy": "gated", "truth_file": "x.json"}})
    with pytest.raises(ConfigError, match="k_cross"):
        load_config(None, {**base, "draft": {"kind": "combined", "policy": "gated", "k_cross": 40}})
    with pytest.raises(ConfigError, match="max_patches"):
        load_config(None, {**base, "draft": {"kind": "combined", "policy": "gated", "max_patches": 9}})
