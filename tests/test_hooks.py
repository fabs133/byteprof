"""HookTimer against a tiny fake byte-LM on CPU (skipped if torch is missing)."""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from byteprof.components import match_components  # noqa: E402
from byteprof.report import render_markdown  # noqa: E402
from byteprof.timing import STEP, CpuClock, HookTimer, summarize  # noqa: E402


class TinyByteLM(nn.Module):
    """Mimics an LTLM step: encoder + decoder every byte, global model every 2nd byte."""

    def __init__(self):
        super().__init__()
        self.local_encoder = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
        self.global_model = nn.Linear(8, 8)
        self.local_decoder = nn.Linear(8, 8)
        self.lm_head = nn.Linear(8, 4)
        self.calls = 0

    def forward(self, x):
        h = self.local_encoder(x)
        if self.calls % 2 == 0:
            h = self.global_model(h)
        self.calls += 1
        return self.lm_head(self.local_decoder(h))


def _timed(model, steps):
    names = [n for n, _ in model.named_modules()]
    patterns = {
        "local_encoder": [r"^local_encoder$"],
        "inner": [r"^local_encoder\.0$"],  # nested -> must be dropped
        "global_model": [r"^global_model$"],
        "local_decoder": [r"^local_decoder$"],
        "lm_head": [r"^lm_head$"],
    }
    matches = match_components(names, patterns)
    mods = [(m.component, model.get_submodule(m.module_name)) for m in matches]
    with HookTimer(model, mods, CpuClock()) as timer:
        timer.start_run(0)
        for _ in range(steps):
            model(torch.randn(1, 8))
        spans = timer.collect()
    return matches, spans


def test_spans_per_step_and_component():
    model = TinyByteLM()
    matches, spans = _timed(model, steps=4)
    assert not any(m.component == "inner" for m in matches)
    step_ids = sorted(s.step for s in spans if s.component == STEP)
    assert step_ids == [0, 1, 2, 3]
    bd = summarize(spans, runs=1, generated_bytes=3, wall_ms=sum(s.ms for s in spans if s.component == STEP))
    assert bd.prefill.steps == 1 and bd.decode.steps == 3
    assert bd.decode.components["local_decoder"].calls == 3
    assert bd.decode.components["global_model"].calls == 1  # steps 1..3 -> only step 2 is even
    assert bd.decode.unattributed_ms >= 0
    md = render_markdown(bd, {"model_id": "tiny"}, baseline_wall_ms=None)
    assert "| local_decoder | 3 |" in md


def test_hooks_removed_after_context():
    model = TinyByteLM()
    _timed(model, steps=1)
    assert not model._forward_hooks and not model._forward_pre_hooks
    assert not model.local_encoder._forward_hooks
