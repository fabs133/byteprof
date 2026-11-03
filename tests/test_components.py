from byteprof.components import ComponentMatch, match_components, unmatched_components

NAMES = [
    "",
    "model",
    "model.local_encoder",
    "model.local_encoder.mlstm",
    "model.boundary_predictor",
    "model.global_model",
    "model.global_model.layers",
    "model.global_model.layers.0",
    "model.local_decoder",
    "model.local_decoder.layers.0",
    "lm_head",
]


def test_maps_each_component_to_its_module():
    patterns = {
        "local_encoder": [r"local_encoder$"],
        "boundary": [r"boundary_predictor$"],
        "global": [r"global_model$"],
        "local_decoder": [r"local_decoder$"],
        "lm_head": [r"^lm_head$"],
    }
    got = match_components(NAMES, patterns)
    assert {(m.component, m.module_name) for m in got} == {
        ("local_encoder", "model.local_encoder"),
        ("boundary", "model.boundary_predictor"),
        ("global", "model.global_model"),
        ("local_decoder", "model.local_decoder"),
        ("lm_head", "lm_head"),
    }


def test_nested_matches_are_dropped_to_avoid_double_counting():
    # "layers" would match inside both global_model and local_decoder.
    patterns = {"global": [r"global_model$"], "layers": [r"layers\.0$"]}
    got = match_components(NAMES, patterns)
    assert ComponentMatch("global", "model.global_model") in got
    assert ComponentMatch("layers", "model.local_decoder.layers.0") in got
    assert ComponentMatch("layers", "model.global_model.layers.0") not in got


def test_first_component_in_config_order_wins():
    patterns = {"a": [r"local"], "b": [r"local_encoder"]}
    got = match_components(NAMES, patterns)
    assert ComponentMatch("a", "model.local_encoder") in got
    assert not any(m.component == "b" for m in got)


def test_unmatched_components_reported():
    patterns = {"global": [r"global_model$"], "ghost": [r"does_not_exist"]}
    assert unmatched_components(match_components(NAMES, patterns), patterns) == ["ghost"]


def test_root_is_never_matched():
    assert match_components([""], {"all": [r".*"]}) == []
