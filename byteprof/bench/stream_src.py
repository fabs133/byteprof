"""CUDA source of the Stage 2 streaming benchmark: the local stack's matrix-vector chain as one persistent kernel.

One decode step of Bolmo's local stack at batch 1 (K = 0 round): encoder layer, out_projection, in_projection,
four decoder layers, final norm + lm_head. Every layer is five dependent phases:
  A  RMSNorm + q, k, v, ogate (+ igate, fgate) projections       (matvec, all-to-all input dependency)
  B  mLSTM state update + output gating                           (small phase: 5 MiB-per-step state traffic)
  C  out projection + residual
  D  RMSNorm + gate/up projections + silu(gate) * up              (pair epilogue)
  E  down projection + residual
Shapes are the real ones (hidden 2048, 16 heads, dk 64, dv 128, mlp 2816, vocab 520). The norms and the mLSTM update are
"modelled": the same data sizes and dependency structure, simplified math (documented in FINDINGS.md); both the
persistent and the per-phase variants run the same device code, and a torch fp32 reference computes the same math.

Kernels per precision Q (0 bf16, 1 int8 per-row scale, 2 int4 group 128 with fp16 scale):
  mega_<q>      persistent, one block per SM, counter barrier between phases, optional cp.async prefetch of the next
                phase's first rows into shared memory before the barrier wait (npre_bytes > 0) and mLSTM state preload
  phase_<q>     one phase per launch (today's style; used in a CUDA graph)
Every spin loop is capped (wall clock) and sets an error flag.
"""

from .sync_src import COMMON

KERNEL = r"""
#include <cuda_fp16.h>
#define THREADS 256
#define NWARP 8
#define MAXMAT 6
#define MAXU 4
#define NUNITS 128
#define KMAX 2816
#ifndef EARLY
#define EARLY 1   // issue the first weight batch before the input vector is loaded
#endif

struct Mat {
  const unsigned char* w;
  const float* rs;     // int8: per-row scales
  float* out;          // output (epi 0)
  int rows;
  int pad;
};
struct Phase {
  int kind, nmat, K, rowstride;
  int epi, has_norm, rows_total, unit;
  const float* in;
  const float* norm_w;
  const float* res;
  float* out0;
  Mat m[MAXMAT];
  float* c; float* n; float* n2; float* mst; float* mst2;   // n, mst double-buffered (parity of the step)
  const float* q; const float* k; const float* v; const float* og; const float* ig; const float* fg;
  float* hout;
};
__device__ int g_phase_size = (int)sizeof(Phase);

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

struct Smem {
  unsigned char* sw;   // staged weight rows
  float* xs;           // input vector (fp32)
  float* xsum;         // per-32 vector sums of xs (int4)
  float* red;          // scratch
};

// ---- input vector prologue: x -> smem (optionally RMSNorm'ed) -----------------------------------------------------
__device__ void load_input(const Phase& ph, const Smem& s, int t) {
  const int K = ph.K;
  const float4* in4 = (const float4*)ph.in;
  float ss = 0.f;
  for (int i = t; i < K / 4; i += THREADS) {
    float4 v = __ldcg(in4 + i);
    ((float4*)s.xs)[i] = v;
    ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  if (ph.has_norm) {
    ss = warp_sum(ss);
    if ((t & 31) == 0) s.red[t >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < NWARP; ++i) tot += s.red[i];
    const float inv = rsqrtf(tot / K + 1e-6f);
    for (int i = t; i < K; i += THREADS) s.xs[i] = s.xs[i] * inv * __ldg(ph.norm_w + i);
  }
  __syncthreads();
}
__device__ void group_sums(const Phase& ph, const Smem& s, int t) {   // int4: sum of x per group of 128
  for (int j = t; j < ph.K / 32; j += THREADS) {   // sum of x over each 32-element vector
    float v = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) { const float4 x4 = *(const float4*)(s.xs + 32 * j + 4 * i); v += x4.x + x4.y + x4.z + x4.w; }
    s.xsum[j] = v;
  }
  __syncthreads();
}

// ---- one row's contribution of a 16-byte weight vector ----------------------------------------------------------------
template <int Q> struct EPV { static constexpr int v = (Q == 0) ? 8 : (Q == 1 ? 16 : 32); };

template <int Q> __device__ __forceinline__ float dot_vec(const uint4& u, const float* xs, int j) {
  const int e0 = j * EPV<Q>::v;
  float a = 0.f;
  if (Q == 0) {
    const unsigned w[4] = {u.x, u.y, u.z, u.w};
    const float4 x0 = *(const float4*)(xs + e0), x1 = *(const float4*)(xs + e0 + 4);
    const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      a = fmaf(__uint_as_float(w[i] << 16), xv[2 * i], a);
      a = fmaf(__uint_as_float(w[i] & 0xffff0000u), xv[2 * i + 1], a);
    }
  } else if (Q == 1) {
    const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float4 xv = *(const float4*)(xs + e0 + 4 * i);
      a = fmaf((float)(int)(signed char)(w[i] & 0xff), xv.x, a);
      a = fmaf((float)(int)(signed char)((w[i] >> 8) & 0xff), xv.y, a);
      a = fmaf((float)(int)(signed char)((w[i] >> 16) & 0xff), xv.z, a);
      a = fmaf((float)(int)(signed char)(w[i] >> 24), xv.w, a);
    }
  } else {
    const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float4 xa = *(const float4*)(xs + e0 + 8 * i), xb = *(const float4*)(xs + e0 + 8 * i + 4);
      const unsigned v = w[i];
      a = fmaf((float)(v & 0xf), xa.x, a);
      a = fmaf((float)((v >> 4) & 0xf), xa.y, a);
      a = fmaf((float)((v >> 8) & 0xf), xa.z, a);
      a = fmaf((float)((v >> 12) & 0xf), xa.w, a);
      a = fmaf((float)((v >> 16) & 0xf), xb.x, a);
      a = fmaf((float)((v >> 20) & 0xf), xb.y, a);
      a = fmaf((float)((v >> 24) & 0xf), xb.z, a);
      a = fmaf((float)(v >> 28), xb.w, a);
    }
  }
  return a;
}

// virtual row -> (matrix, row)
__device__ __forceinline__ void locate(const Phase& ph, int v, int& mi, int& r) {
  if (ph.epi == 2) { mi = v & 1; r = v >> 1; return; }
  r = v;
  mi = 0;
#pragma unroll
  for (int i = 0; i < MAXMAT; ++i) {
    if (i < ph.nmat) {
      if (r < ph.m[i].rows) { mi = i; return; }
      r -= ph.m[i].rows;
    }
  }
}
__device__ __forceinline__ int block_start(int units, int b, int G) { return (int)(((long long)units * b) / G); }

template <int Q> struct Cfg { static constexpr int RB = (Q == 0) ? 2 : 4; static constexpr int NV = (Q == 0) ? 11 : (Q == 1 ? 6 : 3); };

template <int Q>
__device__ void run_matvec(const Phase& ph, const Smem& s, int b, int G, int npre, bool early) {
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  constexpr int RB = Cfg<Q>::RB, NV = Cfg<Q>::NV;
  const int unit = ph.unit;
  const int v0 = block_start(ph.rows_total / unit, b, G) * unit;
  const int v1 = block_start(ph.rows_total / unit, b + 1, G) * unit;
  const int K = ph.K;
  const int nvec = K / EPV<Q>::v;
  // The first batch of weight loads is issued before the input vector is read (weights do not depend on it): the
  // x load, the norm and the weight latency overlap. Every warp runs the first iteration (it holds block barriers).
  bool first = true;
  for (int vi = v0 + warp * RB; first || vi < v1; vi += NWARP * RB) {
    uint4 buf[RB][NV];
    int mi[RB], rr[RB];
    bool ok[RB];
#pragma unroll
    for (int q = 0; q < RB; ++q) {
      ok[q] = (vi + q) < v1;
      mi[q] = 0; rr[q] = 0;
      if (ok[q]) locate(ph, vi + q, mi[q], rr[q]);
    }
#pragma unroll
    for (int q = 0; q < RB; ++q) {
      if (ok[q]) {
        const int local = vi + q - v0;
        const unsigned char* base = (local < npre) ? (s.sw + (size_t)local * ph.rowstride)
                                                   : (ph.m[mi[q]].w + (size_t)rr[q] * ph.rowstride);
#pragma unroll
        for (int i = 0; i < NV; ++i) {
          const int j = lane + 32 * i;
          if (j < nvec) buf[q][i] = (local < npre) ? *(const uint4*)(base + 16 * j) : __ldcs((const uint4*)(base + 16 * j));
        }
      }
    }
    if (first) {
      if (early) { load_input(ph, s, t); if (Q == 2) group_sums(ph, s, t); }
      first = false;
    }
    float y[RB];
#pragma unroll
    for (int q = 0; q < RB; ++q) {
      y[q] = 0.f;
      if (ok[q]) {
        const int local = vi + q - v0;
        const unsigned char* base = (local < npre) ? (s.sw + (size_t)local * ph.rowstride)
                                                   : (ph.m[mi[q]].w + (size_t)rr[q] * ph.rowstride);
        float a = 0.f;
        if (Q == 2) {
          const unsigned short* sc = (const unsigned short*)(base + K / 2);
#pragma unroll
          for (int i = 0; i < NV; ++i) {
            const int j = lane + 32 * i;
            if (j < nvec) {
              const int g = j >> 2;
              const float scale = __half2float(__ushort_as_half(sc[g]));
              a += scale * (dot_vec<2>(buf[q][i], s.xs, j) - 8.f * s.xsum[j]);
            }
          }
        } else {
#pragma unroll
          for (int i = 0; i < NV; ++i) {
            const int j = lane + 32 * i;
            if (j < nvec) a += dot_vec<Q>(buf[q][i], s.xs, j);
          }
        }
        y[q] = warp_sum(a);
      }
    }
    if (lane == 0) {
      if (ph.epi == 2) {
#pragma unroll
        for (int q = 0; q < RB; q += 2) {
          if (ok[q]) {
            float g = y[q], u = y[q + 1];
            if (Q == 1) { g *= ph.m[0].rs[rr[q]]; u *= ph.m[1].rs[rr[q + 1]]; }
            ph.out0[rr[q]] = (g / (1.f + __expf(-g))) * u;
          }
        }
      } else {
#pragma unroll
        for (int q = 0; q < RB; ++q) {
          if (ok[q]) {
            float v = y[q];
            if (Q == 1) v *= ph.m[mi[q]].rs[rr[q]];
            if (ph.epi == 1) ph.out0[rr[q]] = __ldcg(ph.res + rr[q]) + v;
            else ph.m[mi[q]].out[rr[q]] = v;
          }
        }
      }
    }
  }
}

// stage the first rows of the next matvec phase into shared memory (cp.async), return the staged row count
template <int Q>
__device__ int stage_rows(const Phase& ph, const Smem& s, int b, int G, int sw_bytes) {
  const int t = threadIdx.x;
  const int unit = ph.unit;
  const int v0 = block_start(ph.rows_total / unit, b, G) * unit;
  const int v1 = block_start(ph.rows_total / unit, b + 1, G) * unit;
  int n = sw_bytes / ph.rowstride;
  n = min(n, v1 - v0);
  if (unit == 2) n &= ~1;
  const int cpr = ph.rowstride / 16;
  for (int idx = t; idx < n * cpr; idx += THREADS) {
    const int row = idx / cpr, c = idx - row * cpr;
    int mi, r;
    locate(ph, v0 + row, mi, r);
    const unsigned char* src = ph.m[mi].w + (size_t)r * ph.rowstride + 16 * c;
    unsigned dst = (unsigned)__cvta_generic_to_shared(s.sw + (size_t)row * ph.rowstride + 16 * c);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src));
  }
  asm volatile("cp.async.commit_group;");
  return n;
}

// ---- mLSTM state update (phase B) ------------------------------------------------------------------------------------
// unit u = (head h = u / 8, dv slice of 16 = u % 8). State c[h][dk 64][dv 128] fp32, n[h][64], m[h].
__device__ __forceinline__ void mlstm_preload(const Phase& ph, int b, int G, float (&cr)[MAXU][4]) {
  const int t = threadIdx.x, dvl = t & 15, g = t >> 4;
  int ui = 0;
  for (int u = b; u < NUNITS; u += G, ++ui) {
    const int h = u >> 3, dv0 = (u & 7) * 16;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) cr[ui][kk] = ph.c[((size_t)(h * 64 + g * 4 + kk)) * 128 + dv0 + dvl];
  }
}
__device__ void run_mlstm(const Phase& ph, const Smem& s, int b, int G, float (&cr)[MAXU][4], bool preloaded, int par) {
  const float* nsrc = par ? ph.n2 : ph.n; float* ndst = par ? ph.n : ph.n2;
  const float* msrc = par ? ph.mst2 : ph.mst; float* mdst = par ? ph.mst : ph.mst2;
  const int t = threadIdx.x, dvl = t & 15, g = t >> 4;
  if (!preloaded) mlstm_preload(ph, b, G, cr);
  int ui = 0;
  for (int u = b; u < NUNITS; u += G, ++ui) {
    const int h = u >> 3, dv0 = (u & 7) * 16;
    const float igv = __ldcg(ph.ig + h), fgv = __ldcg(ph.fg + h);
    const float mold = msrc[h];
    const float lf = -log1pf(__expf(-fgv));            // logsigmoid(fg)
    const float mnew = fmaxf(lf + mold, igv);
    const float ip = __expf(igv - mnew), fp = __expf(lf + mold - mnew);
    // den = | n_new . q | with n_new = fp * n + ip * k / 8   (every thread recomputes it: no extra barrier)
    float den = 0.f;
    {
      const float4* n4 = (const float4*)(nsrc + h * 64);
      const float4* k4 = (const float4*)(ph.k + h * 64);
      const float4* q4 = (const float4*)(ph.q + h * 64);
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const float4 nn = n4[i], kk = __ldcg(k4 + i), qq = __ldcg(q4 + i);
        den += (fp * nn.x + ip * 0.125f * kk.x) * qq.x + (fp * nn.y + ip * 0.125f * kk.y) * qq.y +
               (fp * nn.z + ip * 0.125f * kk.z) * qq.z + (fp * nn.w + ip * 0.125f * kk.w) * qq.w;
      }
    }
    den = fmaxf(fabsf(den), __expf(-mnew));
    const float vv = __ldcg(ph.v + h * 128 + dv0 + dvl);
    float part = 0.f;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      const int dk = g * 4 + kk;
      const float kd = __ldcg(ph.k + h * 64 + dk) * 0.125f, qd = __ldcg(ph.q + h * 64 + dk);
      const float cn = fp * cr[ui][kk] + ip * kd * vv;
      ph.c[((size_t)(h * 64 + dk)) * 128 + dv0 + dvl] = cn;
      part += cn * qd;
    }
    s.red[t] = part;
    __syncthreads();
    if (t < 16) {
      float num = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) num += s.red[i * 16 + t];
      const float og = __ldcg(ph.og + h * 128 + dv0 + t);
      ph.hout[h * 128 + dv0 + t] = (num / (den + 1e-6f)) / (1.f + __expf(-og));
    }
    if ((u & 7) == 0 && t < 64) ndst[h * 64 + t] = fp * nsrc[h * 64 + t] + ip * 0.125f * __ldcg(ph.k + h * 64 + t);
    if ((u & 7) == 0 && t == 0) mdst[h] = mnew;
    __syncthreads();
  }
}

template <int Q>
__device__ __forceinline__ void run_phase(const Phase& ph, const Smem& s, int b, int G, int npre, float (&cr)[MAXU][4], bool preloaded, int par) {
  if (ph.kind == 0) {
    if (!EARLY) { load_input(ph, s, threadIdx.x); if (Q == 2) group_sums(ph, s, threadIdx.x); }
    run_matvec<Q>(ph, s, b, G, npre, EARLY);
  } else {
    run_mlstm(ph, s, b, G, cr, preloaded, par);
  }
}

#define SMEM_LAYOUT(sw_bytes) \
  extern __shared__ __align__(16) unsigned char smem_raw[]; \
  Smem s; s.sw = smem_raw; s.xs = (float*)(smem_raw + (sw_bytes)); s.xsum = s.xs + KMAX; s.red = s.xsum + 96;

// ---- persistent kernel ----------------------------------------------------------------------------------------------
template <int Q>
__device__ void mega_body(const Phase* phs, int nph, unsigned* ctl, unsigned* counter, int sw_bytes, int prefetch,
                          unsigned long long cap_ns, int skip_block, int skip_phase, unsigned long long* ts) {
  SMEM_LAYOUT(sw_bytes)
  const int G = gridDim.x, b = blockIdx.x, t = threadIdx.x;
  volatile unsigned* err = ctl + 2;
  const unsigned base = ctl[0] * (unsigned)(nph - 1) * (unsigned)G;   // nph - 1 barriers per step
  float cr[MAXU][4];
  bool preloaded = false, preloaded_next = false;
  int npre = 0, npre_next = 0;
  const int par = (int)(ctl[0] & 1u);
  const int half = (sw_bytes / 2) & ~15;
  __shared__ int stop;
  // prefetch: 0 none; 1 stage the next phase's rows at the end of this phase (cp.async in flight during the barrier);
  //           2 double buffer: stage them at the start of this phase (in flight during this phase's streaming)
  for (int p = 0; p < nph; ++p) {
    if (ts && t == 0) ts[((size_t)p * G + b) * 3 + 0] = gtimer();
    Smem sc = s;
    if (prefetch == 2) {
      sc.sw = s.sw + (p & 1) * half;
      if (p + 1 < nph) {
        Smem sn = s;
        sn.sw = s.sw + ((p + 1) & 1) * half;
        const Phase& nx = phs[p + 1];
        if (nx.kind == 0) npre_next = stage_rows<Q>(nx, sn, b, G, half);
        else mlstm_preload(nx, b, G, cr), preloaded_next = true;
      }
    }
    run_phase<Q>(phs[p], sc, b, G, npre, cr, preloaded, par);
    preloaded = preloaded_next; preloaded_next = false;
    npre = npre_next; npre_next = 0;
    __syncthreads();                         // all reads of staged rows / outputs of this phase are done
    if (ts && t == 0) ts[((size_t)p * G + b) * 3 + 1] = gtimer();
    if (p + 1 < nph) {
      const Phase& nx = phs[p + 1];
      if (prefetch == 1) {
        npre = 0; preloaded = false;
        if (nx.kind == 0) npre = stage_rows<Q>(nx, s, b, G, sw_bytes);
        else mlstm_preload(nx, b, G, cr), preloaded = true;
      }
      if (t == 0 && !(b == skip_block && p == skip_phase)) red_rel(counter);
      if (t == 0) { wait_ge<false>(counter, base + (unsigned)(p + 1) * G, err, cap_ns); stop = (int)*err; }
      asm volatile("cp.async.wait_all;" ::: "memory");
      __syncthreads();
      if (ts && t == 0) ts[((size_t)p * G + b) * 3 + 2] = gtimer();
      if (stop) return;       // uniform: thread 0 decided for the whole block
    }
  }
  if (t == 0) {   // the last block out advances the epoch (all blocks have read ctl[0] by now)
    __threadfence();
    if (atomicAdd(ctl + 1, 1u) == (unsigned)G - 1) { ctl[1] = 0; __threadfence(); ctl[0] = ctl[0] + 1; }
  }
}

template <int Q>
__device__ void phase_body(const Phase* phs, int p, int sw_bytes, int par) {
  SMEM_LAYOUT(sw_bytes)
  float cr[MAXU][4];
  run_phase<Q>(phs[p], s, blockIdx.x, gridDim.x, 0, cr, false, par);
}

#define DEFINE_KERNELS(NAME, Q) \
extern "C" __global__ void mega_##NAME(const Phase* phs, int nph, unsigned* ctl, unsigned* counter, int sw_bytes, \
                                       int prefetch, unsigned long long cap_ns, int skip_block, int skip_phase, \
                                       unsigned long long* ts) { \
  mega_body<Q>(phs, nph, ctl, counter, sw_bytes, prefetch, cap_ns, skip_block, skip_phase, ts); } \
extern "C" __global__ void phase_##NAME(const Phase* phs, int p, int sw_bytes, int par) { phase_body<Q>(phs, p, sw_bytes, par); }

DEFINE_KERNELS(bf16, 0)
DEFINE_KERNELS(i8, 1)
DEFINE_KERNELS(i4, 2)

// ---- reference bandwidth kernels ---------------------------------------------------------------------------------------
extern "C" __global__ void read_sum(const uint4* __restrict__ p, size_t n16, float* out) {
  float a = 0.f;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = __ldcs(p + i);
    a += __uint_as_float(v.x) * 1e-30f + __uint_as_float(v.y) * 1e-30f + __uint_as_float(v.z) * 1e-30f + __uint_as_float(v.w) * 1e-30f;
  }
  if (a == 12345.f) out[0] = a;
}
extern "C" __global__ void copy16(const uint4* __restrict__ a, uint4* __restrict__ b, size_t n16) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x)
    b[i] = __ldcs(a + i);
}
"""

SOURCE = COMMON + KERNEL
