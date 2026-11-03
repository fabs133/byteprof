import pytest

from byteprof.timing import STEP, Span, summarize


def _run(run: int) -> list[Span]:
    # step 0 = prefill (10 ms: enc 2, glob 6), steps 1..3 = decode (2 ms each: dec 1.5)
    spans = [
        Span(run, 0, "enc", 2.0),
        Span(run, 0, "glob", 6.0),
        Span(run, 0, STEP, 10.0),
    ]
    for step in (1, 2, 3):
        spans += [Span(run, step, "dec", 1.5), Span(run, step, STEP, 2.0)]
    spans.append(Span(run, 2, "glob", 0.4))  # global runs only on one decode step (patch end)
    return spans


def test_prefill_decode_split_and_unattributed():
    bd = summarize(_run(0) + _run(1), runs=2, generated_bytes=6, wall_ms=40.0)
    assert bd.prefill.steps == 2 and bd.decode.steps == 6
    assert bd.prefill.step_ms == pytest.approx(20.0)
    assert bd.prefill.unattributed_ms == pytest.approx(4.0)
    assert bd.decode.components["dec"].calls == 6
    assert bd.decode.components["glob"].calls == 2
    assert bd.decode.unattributed_ms == pytest.approx(12.0 - 9.0 - 0.8)
    assert bd.outside_step_ms == pytest.approx(40.0 - 20.0 - 12.0)
    assert bd.decode_ms_per_byte == pytest.approx(2.0)
    assert bd.warnings == []


def test_empty_spans_do_not_divide_by_zero():
    bd = summarize([], runs=0, generated_bytes=0, wall_ms=0.0)
    assert bd.decode_ms_per_byte == 0.0
    assert bd.bytes_per_second == 0.0
    assert not bd.step_observed
    assert any("phase split unavailable" in w for w in bd.warnings)


def test_overlapping_components_are_flagged():
    spans = [Span(0, 0, "a", 8.0), Span(0, 0, "b", 8.0), Span(0, 0, STEP, 10.0)]
    bd = summarize(spans, runs=1, generated_bytes=1, wall_ms=10.0)
    assert any("exceeds step time" in w for w in bd.warnings)


def test_calls_outside_step_are_kept_separately():
    spans = [Span(0, -1, "enc", 1.0), Span(0, 0, STEP, 2.0)]
    bd = summarize(spans, runs=1, generated_bytes=1, wall_ms=3.0)
    assert bd.outside_step_components["enc"].calls == 1
    assert "enc" not in bd.prefill.components
