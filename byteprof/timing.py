"""Forward-hook timing of model components.

Design:
  * A `Clock` produces opaque marks and resolves mark pairs to milliseconds.
    `CudaEventClock` records CUDA events (async, no per-call sync), `CpuClock`
    uses perf_counter (used for tests and CPU runs).
  * `HookTimer` registers pre/post forward hooks on the step module and on each
    matched component module. Every step-module forward is one "step"; within a
    run, step 0 is the prefill and later steps are decode steps.
  * `summarize` is a pure function from spans to a breakdown.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol

STEP = "__step__"  # pseudo-component for the step module itself


class Clock(Protocol):
    def mark(self) -> Any: ...
    def resolve(self, start: Any, end: Any) -> float: ...
    def synchronize(self) -> None: ...


class CpuClock:
    def mark(self) -> float:
        return time.perf_counter()

    def resolve(self, start: float, end: float) -> float:
        return (end - start) * 1000.0

    def synchronize(self) -> None:
        pass


class CudaEventClock:
    def __init__(self) -> None:
        import torch

        self._torch = torch

    def mark(self) -> Any:
        ev = self._torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev

    def resolve(self, start: Any, end: Any) -> float:
        return float(start.elapsed_time(end))

    def synchronize(self) -> None:
        self._torch.cuda.synchronize()


@dataclass(frozen=True)
class Span:
    run: int
    step: int  # -1 = outside any step-module forward
    component: str
    ms: float


class HookTimer:
    """Collects timed spans via forward hooks. Use as a context manager."""

    def __init__(self, step_module: Any, components: list[tuple[str, Any]], clock: Clock) -> None:
        self._step_module = step_module
        self._components = components  # (component name, nn.Module); a component may span modules
        self._clock = clock
        self._handles: list[Any] = []
        self._pending: list[tuple[int, int, str, Any, Any]] = []
        self._open: dict[int, list[Any]] = defaultdict(list)
        self._run = 0
        self._step = -1

    # -- hook management -------------------------------------------------
    def __enter__(self) -> HookTimer:
        self._attach(self._step_module, STEP)
        for name, module in self._components:
            self._attach(module, name)
        return self

    def __exit__(self, *exc: object) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _attach(self, module: Any, component: str) -> None:
        def pre(mod: Any, _args: Any) -> None:
            if component == STEP:
                self._step += 1
            self._open[id(mod)].append(self._clock.mark())

        def post(mod: Any, _args: Any, _out: Any) -> None:
            stack = self._open[id(mod)]
            if not stack:  # hook attached mid-forward; ignore
                return
            start = stack.pop()
            step = self._step if (component == STEP or self._step >= 0) else -1
            self._pending.append((self._run, step, component, start, self._clock.mark()))

        self._handles.append(module.register_forward_pre_hook(pre))
        self._handles.append(module.register_forward_hook(post))

    # -- run lifecycle ---------------------------------------------------
    def start_run(self, run: int) -> None:
        self._run = run
        self._step = -1

    def collect(self) -> list[Span]:
        self._clock.synchronize()
        spans = [
            Span(run, step, comp, self._clock.resolve(s, e))
            for run, step, comp, s, e in self._pending
        ]
        self._pending.clear()
        return spans


# ---------------------------------------------------------------------------
# Pure aggregation
# ---------------------------------------------------------------------------


@dataclass
class ComponentStats:
    calls: int = 0
    total_ms: float = 0.0

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0


@dataclass
class PhaseBreakdown:
    step_ms: float = 0.0
    steps: int = 0
    components: dict[str, ComponentStats] = field(default_factory=dict)

    @property
    def attributed_ms(self) -> float:
        return sum(c.total_ms for c in self.components.values())

    @property
    def unattributed_ms(self) -> float:
        """Step time not covered by any component (glue, sampling inside forward, launches)."""
        return self.step_ms - self.attributed_ms


@dataclass
class Breakdown:
    runs: int
    generated_bytes: int
    wall_ms: float  # total wall time of generate() calls (hooked runs)
    prefill: PhaseBreakdown
    decode: PhaseBreakdown
    outside_step_components: dict[str, ComponentStats]
    step_observed: bool
    warnings: list[str]

    @property
    def outside_step_ms(self) -> float:
        """Wall time not inside any step forward: generate() loop, sampling, host overhead."""
        return self.wall_ms - self.prefill.step_ms - self.decode.step_ms

    @property
    def decode_ms_per_byte(self) -> float:
        return self.decode.step_ms / self.generated_bytes if self.generated_bytes else 0.0

    @property
    def bytes_per_second(self) -> float:
        return self.generated_bytes / (self.wall_ms / 1000.0) if self.wall_ms > 0 else 0.0


def summarize(spans: list[Span], *, runs: int, generated_bytes: int, wall_ms: float) -> Breakdown:
    prefill, decode = PhaseBreakdown(), PhaseBreakdown()
    outside: dict[str, ComponentStats] = {}
    step_observed = any(s.component == STEP for s in spans)
    for s in spans:
        if s.component == STEP:
            phase = prefill if s.step == 0 else decode
            phase.step_ms += s.ms
            phase.steps += 1
            continue
        if s.step < 0:
            bucket = outside
        else:
            bucket = (prefill if s.step == 0 else decode).components
        stats = bucket.setdefault(s.component, ComponentStats())
        stats.calls += 1
        stats.total_ms += s.ms

    warnings: list[str] = []
    if not step_observed:
        warnings.append(
            "step module forward was never called: phase split unavailable. "
            "Set `step_module` in the config to the module whose forward runs once per step."
        )
    for label, phase in (("prefill", prefill), ("decode", decode)):
        if phase.unattributed_ms < -0.05 * max(phase.step_ms, 1e-9):
            warnings.append(
                f"{label}: component time exceeds step time by "
                f"{-phase.unattributed_ms:.2f} ms - components probably overlap or run "
                "asynchronously across steps; check the [components] patterns."
            )
    if outside:
        warnings.append(
            "some component calls happened outside the step module "
            f"({', '.join(sorted(outside))}); they are reported separately."
        )
    return Breakdown(runs, generated_bytes, wall_ms, prefill, decode, outside, step_observed, warnings)
