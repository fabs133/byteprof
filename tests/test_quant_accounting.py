import pytest

from byteprof.quant.accounting import (
    Calibration, Component, Linear, Variant, bytes_per_output_byte, component_bytes, linear_bytes, predict_ms,
)
from byteprof.quant.select import group_of, should_quantize


def toy():
    # local: one 8x256 linear read every byte; global: one 16x256 linear read every 4th byte;
    # head: 4x256 (never quantized); embeddings: 2 rows of 256 per byte
    return [
        Component("enc", 1.0, (Linear(8, 256, "local"),), other_params=10, kind="local"),
        Component("glob", 0.25, (Linear(16, 256, "global"),), kind="global"),
        Component("head", 1.0, (Linear(4, 256, None),), kind="head"),
        Component("emb", 1.0, (), other_params=512, kind="embed"),
    ]


def test_linear_bytes_schemes():
    lin = Linear(8, 256, "local")
    assert linear_bytes(lin, None) == 8 * 256 * 2
    assert linear_bytes(lin, "int8_wo") == 8 * 256 + 8 * 2
    # int4, group 128: 2 groups per row, 4 bytes (bf16 scale + zero) per group per row
    assert linear_bytes(lin, "int4_wo", 128) == 8 * 256 * 0.5 + 8 * 2 * 4
    with pytest.raises(ValueError):
        linear_bytes(Linear(8, 100, None), "int4_wo", 128)


def test_bf16_totals_on_toy():
    b = bytes_per_output_byte(toy())
    assert b["enc"] == 8 * 256 * 2 + 10 * 2
    assert b["glob"] == 16 * 256 * 2 * 0.25
    assert b["head"] == 4 * 256 * 2
    assert b["emb"] == 1024
    assert b["total"] == pytest.approx(b["enc"] + b["glob"] + b["head"] + b["emb"])
    assert b["kind:local"] == b["enc"]


def test_target_scopes_and_never_head_or_embeddings():
    comps = toy()
    base = bytes_per_output_byte(comps)
    local = bytes_per_output_byte(comps, Variant("l", "local", "int8_wo"))
    glob = bytes_per_output_byte(comps, Variant("g", "global", "int8_wo"))
    both = bytes_per_output_byte(comps, Variant("a", "all", "int4_wo"))
    assert local["glob"] == base["glob"] and glob["enc"] == base["enc"]
    assert local["enc"] == 8 * 256 + 16 + 20  # int8 weights + scales + bf16 'other'
    for v in (local, glob, both):
        assert v["head"] == base["head"] and v["emb"] == base["emb"]
    assert both["glob"] == 0.25 * (16 * 256 * 0.5 + 16 * 2 * 4)
    assert component_bytes(comps[0], None) == base["enc"]


def test_predict_ms_proportional_with_constant():
    comps = toy()
    cal = Calibration({"local": 2.0, "global": 6.0}, const_ms=1.0)
    assert cal.total_ms == 9.0
    v = Variant("l", "local", "int8_wo")
    ratio = (8 * 256 + 16 + 20) / (8 * 256 * 2 + 20)
    assert predict_ms(comps, cal, v) == pytest.approx(1.0 + 2.0 * ratio + 6.0)
    assert predict_ms(comps, cal, Variant("none")) == pytest.approx(9.0)


def test_select_local_exactly():
    enc = "model.local_encoder.layers.0."
    dec = "model.local_decoder.layers.3."
    yes = [enc + s for s in ("xlstm.q", "xlstm.k", "xlstm.v", "xlstm.ogate_preact", "xlstm.out_proj",
                              "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
    yes += [dec + s for s in ("xlstm.q", "mlp.down_proj")]
    yes += ["model.local_encoder.out_projection", "model.local_decoder.in_projection"]
    no = ["lm_head", "model.local_encoder.byte_embedding", "model.local_encoder.subword_embedding",
          "model.local_encoder.boundary_predictor_module.q_proj_layer", enc + "xlstm.igate_preact",
          enc + "xlstm.fgate_preact", "model.local_decoder.layers",
          "model.local_encoder.layers.0.mlp", enc + "pre_xlstm_layernorm"]
    for n in yes:
        assert should_quantize(n, "local") and not should_quantize(n, "global") and should_quantize(n, "all"), n
        assert not should_quantize(n, "none")
    assert not should_quantize("model.layers.0.self_attn.q_proj", "local")  # global layers: not local
    for n in no:
        assert not any(should_quantize(n, t) for t in ("none", "local", "global", "all")), n


def test_select_global_exactly():
    yes = [f"model.layers.{i}.{s}" for i in (0, 15) for s in (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
    for n in yes:
        assert group_of(n) == "global" and should_quantize(n, "global") and not should_quantize(n, "local")
    for n in ("model.layers.0.self_attn.q_norm", "model.layers.0", "model.embed_tokens", "lm_head",
              "model.layers.0.self_attn"):
        assert group_of(n) is None
    with pytest.raises(ValueError):
        should_quantize("lm_head", "everything")


def test_olmo_target_filter_exact():
    from byteprof.quant.select import olmo_group_of, olmo_should_quantize, selects

    blocks = [f"model.layers.{i}.{s}" for i in (0, 15) for s in (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
    never = ["model.embed_tokens", "model.norm", "model.layers.0.self_attn.q_norm",
             "model.layers.0.post_attention_layernorm", "model.layers.0.mlp", "model.layers", "lm_head.weight",
             "model.local_encoder.layers.0.xlstm.q"]
    for n in blocks:
        assert olmo_group_of(n) == "blocks"
        assert olmo_should_quantize(n, "blocks") and olmo_should_quantize(n, "blocks+head")
        assert not olmo_should_quantize(n, "none")
    assert olmo_group_of("lm_head") == "head"
    assert olmo_should_quantize("lm_head", "blocks+head") and not olmo_should_quantize("lm_head", "blocks")
    assert not olmo_should_quantize("lm_head", "none")
    for n in never:
        assert not any(olmo_should_quantize(n, t) for t in ("none", "blocks", "blocks+head")), n
    with pytest.raises(ValueError):
        olmo_should_quantize("lm_head", "all")
    with pytest.raises(ValueError):
        should_quantize("model.layers.0.mlp.up_proj", "blocks")  # Bolmo filter rejects OLMo targets
    assert selects("blocks+head", "head") and not selects("all", "head") and not selects("none", "blocks")


def test_olmo_accounting_blocks_vs_head():
    from byteprof.quant.accounting import olmo_components

    lin = {"model.layers.0.mlp.up_proj": (8192, 2048), "lm_head": (100352, 2048)}
    comps = olmo_components(lin, {}, 4.0, 2048, 100352)
    base = bytes_per_output_byte(comps)
    blocks = bytes_per_output_byte(comps, Variant("b", "blocks", "int4_wo"))
    both = bytes_per_output_byte(comps, Variant("bh", "blocks+head", "int4_wo"))
    assert blocks["lm_head"] == base["lm_head"] and blocks["layers"] < base["layers"] * 0.3
    assert both["lm_head"] < base["lm_head"] * 0.3 and both["layers"] == blocks["layers"]


def test_mixed_scheme_variant_local_int4_global_int8():
    comps = toy()
    v = Variant("l4g8", "all", "int4_wo", 128, global_scheme="int8_wo")
    b = bytes_per_output_byte(comps, v)
    assert b["enc"] == 8 * 256 * 0.5 + 8 * 2 * 4 + 20          # local: int4, group 128
    assert b["glob"] == 0.25 * (16 * 256 + 16 * 2)             # global: int8 + per-row bf16 scale
    assert b["head"] == bytes_per_output_byte(comps)["head"]


def test_config_accepts_per_group_schemes(tmp_path):
    from byteprof.config import ConfigError, load_config

    p = tmp_path / "c.toml"
    p.write_text('[quantize]\ntarget = "all"\nscheme = "int4_wo"\nglobal_scheme = "int8_wo"\n', encoding="utf-8")
    assert load_config(p).quantize["global_scheme"] == "int8_wo"
    p.write_text('[quantize]\ntarget = "all"\nscheme = "int4_wo"\nglobal_scheme = "fp8"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)
    p.write_text('[quantize]\ntarget = "blocks+head"\nscheme = "int4_wo"\n', encoding="utf-8")
    assert load_config(p).quantize["target"] == "blocks+head"
