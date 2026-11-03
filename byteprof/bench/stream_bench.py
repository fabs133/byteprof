"""Stage 2 of TASK_megakernel_theory.md: the local stack's matrix-vector chain as per-phase kernels in a CUDA graph vs a
persistent kernel (without / with prefetch of the next phase's weights). Weights are random but have the real shapes.
"""

from __future__ import annotations

import ctypes as C
import statistics
from dataclasses import dataclass, field

import torch

from . import cuda_rt
from .stream_src import SOURCE

D, HEADS, DK, DV, MLP, VOCAB = 2048, 16, 64, 128, 2816, 520
NLAYER = 5          # 1 encoder + 4 decoder
SW_BYTES = 80 * 1024
SMEM_EXTRA = (2816 + 96 + 256) * 4
CAP_NS = 20_000_000
QNAMES = {"bf16": "bf16", "int8": "i8", "int4": "i4"}


class CMat(C.Structure):
    _fields_ = [("w", C.c_uint64), ("rs", C.c_uint64), ("out", C.c_uint64), ("rows", C.c_int), ("pad", C.c_int)]


class CPhase(C.Structure):
    _fields_ = ([("kind", C.c_int), ("nmat", C.c_int), ("K", C.c_int), ("rowstride", C.c_int), ("epi", C.c_int),
                 ("has_norm", C.c_int), ("rows_total", C.c_int), ("unit", C.c_int), ("in_", C.c_uint64),
                 ("norm_w", C.c_uint64), ("res", C.c_uint64), ("out0", C.c_uint64), ("m", CMat * 6)] +
                [(n, C.c_uint64) for n in ("c", "n", "n2", "mst", "mst2", "q", "k", "v", "og", "ig", "fg", "hout")])


_MOD = None


def module() -> cuda_rt.Module:
    global _MOD
    if _MOD is None:
        _MOD = cuda_rt.Module(SOURCE)
        p, n = _MOD.global_ptr("g_phase_size")
        v = torch.zeros(1, dtype=torch.int32, device="cuda")
        # read the device int through a tiny copy
        tmp = (C.c_int * 1)()
        cuda_rt._libs()[1].cuMemcpyDtoH_v2(tmp, C.c_uint64(p), C.c_size_t(4))
        assert tmp[0] == C.sizeof(CPhase), (tmp[0], C.sizeof(CPhase))
    return _MOD


def row_stride(q: str, K: int) -> int:
    if q == "bf16":
        return 2 * K
    if q == "int8":
        return K
    return (K // 2 + 2 * (K // 128) + 15) // 16 * 16


def quantize(w: torch.Tensor, q: str):
    """w: fp32 [rows, K] -> (raw uint8 [rows, rowstride], per-row scale or None, dequantized fp32 or None)."""
    rows, K = w.shape
    if q == "bf16":
        wb = w.bfloat16()
        return wb.view(torch.uint8).reshape(rows, 2 * K).contiguous(), None, wb.float()
    if q == "int8":
        rs = (w.abs().amax(1) / 127).clamp_min(1e-8)
        qi = torch.round(w / rs[:, None]).clamp(-127, 127).to(torch.int8)
        return qi.view(torch.uint8).contiguous(), rs.float().contiguous(), qi.float() * rs[:, None]
    wg = w.view(rows, K // 128, 128)
    sc = (wg.abs().amax(-1) / 7).clamp_min(1e-6).half()
    nib = (torch.round(wg / sc.float()[..., None]).clamp(-8, 7) + 8).to(torch.uint8).view(rows, K)
    packed = nib[:, 0::2] | (nib[:, 1::2] << 4)
    scb = sc.view(torch.uint8).reshape(rows, 2 * (K // 128))
    stride = row_stride("int4", K)
    raw = torch.zeros(rows, stride, dtype=torch.uint8, device=w.device)
    raw[:, : K // 2] = packed
    raw[:, K // 2: K // 2 + scb.shape[1]] = scb
    deq = ((nib.float() - 8).view(rows, K // 128, 128) * sc.float()[..., None]).view(rows, K)
    return raw, None, deq


@dataclass
class QMat:
    raw: torch.Tensor
    rs: torch.Tensor | None
    deq: torch.Tensor | None
    rows: int
    K: int


def make_mat(rows: int, K: int, q: str, gen: torch.Generator, keep_deq: bool) -> QMat:
    w = torch.randn(rows, K, device="cuda", generator=gen) / (K ** 0.5)
    raw, rs, deq = quantize(w, q)
    return QMat(raw, rs, deq if keep_deq else None, rows, K)


@dataclass
class Stack:
    q: str
    keep_deq: bool = False
    seed: int = 0
    layers: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    buf: dict = field(default_factory=dict)

    def __post_init__(self):
        g = torch.Generator(device="cuda")
        g.manual_seed(self.seed)
        mk = lambda r, k: make_mat(r, k, self.q, g, self.keep_deq)  # noqa: E731
        vec = lambda n: (1 + 0.1 * torch.randn(n, device="cuda", generator=g)).float()  # noqa: E731
        for _ in range(NLAYER):
            self.layers.append(dict(
                wq=mk(HEADS * DK, D), wk=mk(HEADS * DK, D), wv=mk(HEADS * DV, D), wog=mk(HEADS * DV, D),
                wig=mk(HEADS, D), wfg=mk(HEADS, D), wo=mk(D, D), wgate=mk(MLP, D), wup=mk(MLP, D), wdown=mk(D, MLP),
                g1=vec(D), g2=vec(D)))
        self.extra = dict(wpo=mk(D, D), wpi=mk(D, D), wlm=mk(VOCAB, D), gpo=vec(D), gf=vec(D))
        f = lambda n: torch.zeros(n, device="cuda", dtype=torch.float32)  # noqa: E731
        b = self.buf
        for name, n in dict(x_in=D, x=D, q=HEADS * DK, k=HEADS * DK, v=HEADS * DV, og=HEADS * DV, ig=HEADS, fg=HEADS,
                            h=HEADS * DV, a=MLP, y1=D, e=D, logits=VOCAB).items():
            b[name] = f(n)
        b["x_in"].copy_(torch.randn(D, device="cuda", generator=g))
        b["e"].copy_(0.1 * torch.randn(D, device="cuda", generator=g))
        self.state = [dict(c=f(HEADS * DK * DV), n=f(HEADS * DK), n2=f(HEADS * DK), m=f(HEADS), m2=f(HEADS))
                      for _ in range(NLAYER)]
        self.ctl = torch.zeros(8, dtype=torch.int32, device="cuda")
        self.counter = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.reset_state()
        self.phases = self._phases()
        self.nph = len(self.phases)
        arr = (CPhase * self.nph)(*self.phases)
        self.phs_dev = torch.frombuffer(bytearray(arr), dtype=torch.uint8).cuda()
        self.sw_bytes = SW_BYTES
        self.weight_bytes = self._weight_bytes()

    def reset_state(self) -> None:
        for s in self.state:
            s["c"].zero_()
            s["n"].zero_(); s["n2"].zero_()
            s["m"].fill_(-3.0); s["m2"].fill_(-3.0)
        self.ctl.zero_()
        self.counter.zero_()

    def _all_mats(self):
        for L in self.layers:
            for k, v in L.items():
                if isinstance(v, QMat):
                    yield v
        for v in self.extra.values():
            if isinstance(v, QMat):
                yield v

    def _weight_bytes(self) -> int:
        return sum(m.raw.numel() for m in self._all_mats()) + sum(m.rs.numel() * 4 for m in self._all_mats() if m.rs is not None)

    # ---- phase table ------------------------------------------------------------------------------------------------
    def _cmat(self, m: QMat, out: torch.Tensor | None) -> CMat:
        return CMat(m.raw.data_ptr(), m.rs.data_ptr() if m.rs is not None else 0, out.data_ptr() if out is not None else 0,
                    m.rows, 0)

    def _matvec(self, mats, outs, K, in_, norm_w=None, epi=0, res=None, out0=None) -> CPhase:
        ph = CPhase()
        ph.kind, ph.nmat, ph.K, ph.rowstride = 0, len(mats), K, row_stride(self.q, K)
        ph.epi, ph.has_norm = epi, 1 if norm_w is not None else 0
        ph.unit = 2 if epi == 2 else 1
        ph.rows_total = sum(m.rows for m in mats)
        ph.in_ = in_.data_ptr()
        ph.norm_w = norm_w.data_ptr() if norm_w is not None else 0
        ph.res = res.data_ptr() if res is not None else 0
        ph.out0 = out0.data_ptr() if out0 is not None else 0
        for i, (m, o) in enumerate(zip(mats, outs)):
            ph.m[i] = self._cmat(m, o)
        return ph

    def _mlstm(self, st) -> CPhase:
        b = self.buf
        ph = CPhase()
        ph.kind = 1
        ph.c, ph.n, ph.n2, ph.mst, ph.mst2 = (st["c"].data_ptr(), st["n"].data_ptr(), st["n2"].data_ptr(), st["m"].data_ptr(),
                                              st["m2"].data_ptr())
        ph.q, ph.k, ph.v, ph.og, ph.ig, ph.fg, ph.hout = (b[n].data_ptr() for n in ("q", "k", "v", "og", "ig", "fg", "h"))
        return ph

    def _phases(self) -> list[CPhase]:
        b = self.buf
        out: list[CPhase] = []

        def layer(i: int, xin: torch.Tensor) -> None:
            L, st = self.layers[i], self.state[i]
            out.append(self._matvec([L["wq"], L["wk"], L["wv"], L["wog"], L["wig"], L["wfg"]],
                                    [b["q"], b["k"], b["v"], b["og"], b["ig"], b["fg"]], D, xin, norm_w=L["g1"]))
            out.append(self._mlstm(st))
            out.append(self._matvec([L["wo"]], [None], D, b["h"], epi=1, res=xin, out0=b["x"]))
            out.append(self._matvec([L["wgate"], L["wup"]], [None, None], D, b["x"], norm_w=L["g2"], epi=2, out0=b["a"]))
            out.append(self._matvec([L["wdown"]], [None], MLP, b["a"], epi=1, res=b["x"], out0=b["x"]))

        layer(0, b["x_in"])
        out.append(self._matvec([self.extra["wpo"]], [b["y1"]], D, b["x"], norm_w=self.extra["gpo"]))
        out.append(self._matvec([self.extra["wpi"]], [None], D, b["y1"], epi=1, res=b["e"], out0=b["x"]))
        for i in range(1, NLAYER):
            layer(i, b["x"])
        out.append(self._matvec([self.extra["wlm"]], [b["logits"]], D, b["x"], norm_w=self.extra["gf"]))
        return out

    # ---- launches ---------------------------------------------------------------------------------------------------
    def smem(self) -> int:
        return self.sw_bytes + SMEM_EXTRA

    def mega(self, prefetch: bool, grid: int | None = None, cap_ns: int = CAP_NS, skip_block: int = -1,
             skip_phase: int = -1, ts: torch.Tensor | None = None) -> None:
        mod = module()
        f = mod.func(f"mega_{QNAMES[self.q]}", self.smem())
        grid = grid or torch.cuda.get_device_properties(0).multi_processor_count
        assert grid <= cuda_rt.occupancy(f, 256, self.smem()) * torch.cuda.get_device_properties(0).multi_processor_count, \
            "persistent grid would not be co-resident"
        cuda_rt.launch(f, grid, 256, [self.phs_dev, self.nph, self.ctl, self.counter, self.sw_bytes, int(prefetch),
                                      C.c_uint64(cap_ns), skip_block, skip_phase,
                                      C.c_uint64(ts.data_ptr() if ts is not None else 0)], smem=self.smem())

    def per_phase(self, grid: int) -> None:
        mod = module()
        f = mod.func(f"phase_{QNAMES[self.q]}", SMEM_EXTRA)   # no staging area: normal kernels keep full occupancy
        for p in range(self.nph):
            cuda_rt.launch(f, grid, 256, [self.phs_dev, p, 0, 0], smem=SMEM_EXTRA)

    def graph_of(self, fn) -> torch.cuda.CUDAGraph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        return g

    # ---- reference --------------------------------------------------------------------------------------------------
    def reference(self, steps: int = 1) -> dict:
        """Same math in fp32 torch from the dequantized weights; fresh state (as after reset_state), `steps` steps."""
        assert self.keep_deq
        b = self.buf

        def rms(x, g):
            return x * torch.rsqrt((x * x).mean() + 1e-6) * g

        def layer(i, x):
            L = self.layers[i]
            hn = rms(x, L["g1"])
            q, k, v, og = (L[n].deq @ hn for n in ("wq", "wk", "wv", "wog"))
            ig, fg = L["wig"].deq @ hn, L["wfg"].deq @ hn
            m0, c0, n0 = carry[i]["m"], carry[i]["c"], carry[i]["n"]
            lf = -torch.log1p(torch.exp(-fg))
            mnew = torch.maximum(lf + m0, ig)
            ip, fp = torch.exp(ig - mnew), torch.exp(lf + m0 - mnew)
            kd = k.view(HEADS, DK) * 0.125
            n_new = fp[:, None] * n0 + ip[:, None] * kd
            c_new = fp[:, None, None] * c0 + ip[:, None, None] * kd[:, :, None] * v.view(HEADS, DV)[:, None, :]
            q2 = q.view(HEADS, DK)
            num = torch.einsum("hkd,hk->hd", c_new, q2)
            den = torch.maximum((n_new * q2).sum(1).abs(), torch.exp(-mnew))[:, None]
            h = (num / (den + 1e-6)) / (1 + torch.exp(-og.view(HEADS, DV)))
            x1 = x + L["wo"].deq @ h.flatten()
            hn2 = rms(x1, L["g2"])
            gg, uu = L["wgate"].deq @ hn2, L["wup"].deq @ hn2
            a = (gg / (1 + torch.exp(-gg))) * uu
            return x1 + L["wdown"].deq @ a, dict(c=c_new, n=n_new, m=mnew)

        carry = [dict(c=torch.zeros(HEADS, DK, DV, device="cuda"), n=torch.zeros(HEADS, DK, device="cuda"),
                      m=torch.full((HEADS,), -3.0, device="cuda")) for _ in range(NLAYER)]
        for _ in range(steps):
            x, sts = b["x_in"].clone(), []
            x, s = layer(0, x); sts.append(s)
            y1 = self.extra["wpo"].deq @ rms(x, self.extra["gpo"])
            x = b["e"] + self.extra["wpi"].deq @ y1
            for i in range(1, NLAYER):
                x, s = layer(i, x); sts.append(s)
            carry = sts
        logits = self.extra["wlm"].deq @ rms(x, self.extra["gf"])
        return dict(x=x, logits=logits, states=[dict(c=s["c"].flatten(), n=s["n"].flatten(), m=s["m"]) for s in sts])

    def fresh_outputs(self, steps: int = 1) -> dict:
        """Outputs after `steps` persistent-kernel steps (n, m live in the buffer the last step wrote: parity)."""
        nk, mk = ("n2", "m2") if steps % 2 == 1 else ("n", "m")
        return dict(x=self.buf["x"].clone(), logits=self.buf["logits"].clone(),
                    states=[dict(c=s["c"].clone(), n=s[nk].clone(), m=s[mk].clone()) for s in self.state])


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


def compare(out: dict, ref: dict) -> dict:
    r = dict(x=rel_err(out["x"], ref["x"]), logits=rel_err(out["logits"], ref["logits"]))
    r["c"] = max(rel_err(o["c"], t["c"]) for o, t in zip(out["states"], ref["states"]))
    r["n"] = max(rel_err(o["n"], t["n"]) for o, t in zip(out["states"], ref["states"]))
    return r


def time_graph(g: torch.cuda.CUDAGraph, reps: int = 30, trials: int = 9) -> float:
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(trials):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / reps)
    return statistics.median(ts)


def copy_bandwidth(nbytes: int = 256 << 20) -> dict:
    """Measured bandwidth (GB/s): read+write copy (the task's reference, counted r+w) and a read-only sum."""
    mod = module()
    n16 = nbytes // 16
    a = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    b = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    out = torch.zeros(1, device="cuda")
    fc, fr = mod.func("copy16"), mod.func("read_sum")
    res = {}
    for name, fn, factor in (("copy_rw", lambda: cuda_rt.launch(fc, 38 * 8, 256, [a, b, C.c_uint64(n16)]), 2),
                             ("read_only", lambda: cuda_rt.launch(fr, 38 * 8, 256, [a, C.c_uint64(n16), out]), 1)):
        g = torch.cuda.CUDAGraph()
        fn(); torch.cuda.synchronize()
        with torch.cuda.graph(g):
            fn()
        us = time_graph(g, reps=10)
        res[name + "_GBs"] = factor * nbytes / us / 1e3
    return res
