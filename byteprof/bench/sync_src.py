"""CUDA source of the Stage 1 synchronization microbenchmark (compiled at runtime with NVRTC).

One persistent block per SM runs a chain of N dependent phases. Each phase: every thread writes its slot of a
per-phase value buffer, the block signals completion with a release-increment, and the next phase waits.
Every spin loop has a wall-clock cap (globaltimer); on timeout it sets *err and all blocks leave (no hang,
so no TDR). Counters are never reset inside the kernel: the host passes `base` (the counter value already
reached before this launch).
"""

COMMON = r"""
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t;
}
__device__ __forceinline__ unsigned ld_acq(const unsigned* p) {
  unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ unsigned ld_rlx(const unsigned* p) {
  unsigned v; asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void red_rel(unsigned* p) {
  asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(p) : "memory");
}
// Spin until *p >= target (wrap-safe). Returns false on timeout or when another block already failed.
template <bool RLX>
__device__ __noinline__ bool wait_ge(const unsigned* p, unsigned target, volatile unsigned* err, unsigned long long cap_ns) {
  unsigned long long t0 = 0;
  unsigned it = 0;
  while ((int)((RLX ? ld_rlx(p) : ld_acq(p)) - target) < 0) {
    if ((++it & 127u) == 0) {
      if (*err) return false;
      unsigned long long t = gtimer();
      if (t0 == 0) t0 = t;
      else if (t - t0 > cap_ns) { atomicExch((unsigned*)err, 1u); return false; }
    }
  }
  if (RLX) asm volatile("fence.acq_rel.gpu;" ::: "memory");
  return true;
}
"""

KERNELS = r"""
extern "C" __global__ void sync_chain(unsigned* counters, float* vals, unsigned* err, unsigned* bad,
                                     int nphase, int mode, int nprod, int work, unsigned base,
                                     unsigned long long cap_ns, int skip_block, int skip_phase, int skip_wait_block, int impl) {
  // impl bit1: relaxed polling + one acq_rel fence after the wait; bit2: skip the publication read-back (pure sync cost).
  // impl bit0: skip the extra __threadfence before the release-add (red.release already orders the block's writes
  // through the preceding __syncthreads); mode 0: all-blocks barrier on one counter; 1: dataflow (wait for nprod ring neighbours' flags);
  // 3: no synchronization (work only)
  const int G = gridDim.x, b = blockIdx.x, t = threadIdx.x, T = blockDim.x;
  unsigned* ctr = counters;            // counters[0]: barrier counter, counters[64 + block]: per-block flags
  unsigned* flag = counters + 64;
  float acc = 0.f;
  for (int p = 0; p < nphase; ++p) {
    float v = (float)(p + 1);
    for (int w = 0; w < work; ++w) v = v * 1.0000001f + 1e-7f;   // dependent chain: tiny, tunable work
    vals[((size_t)p * G + b) * T + t] = (p + 1) + (v - v);
    __syncthreads();
    if (t == 0) {
      if (mode != 3) {
        if (!(impl & 1)) __threadfence();
        if (!(b == skip_block && p == skip_phase)) {
          if (mode == 0) red_rel(ctr); else red_rel(flag + b);
        }
      }
    }
    if (mode == 3) continue;
    // wait for the phase's inputs
    bool ok = true;
    if (t == 0 && b != skip_wait_block) {
      if (mode == 0) ok = (impl & 2) ? wait_ge<true>(ctr, base + (unsigned)(p + 1) * G, err, cap_ns) : wait_ge<false>(ctr, base + (unsigned)(p + 1) * G, err, cap_ns);
    }
    if (mode == 1 && t < nprod && b != skip_wait_block) {
      int src = ((b - 1 - t) % G + G) % G;
      ok = (impl & 2) ? wait_ge<true>(flag + src, base + (unsigned)(p + 1), err, cap_ns) : wait_ge<false>(flag + src, base + (unsigned)(p + 1), err, cap_ns);
    }
    __syncthreads();
    if (*(volatile unsigned*)err) return;
    if (impl & 4) continue;   // sync only: skip the publication read-back
    // read one producer value back and check it is from this phase (publication check)
    int src = (mode == 0) ? (b + 1) % G : ((b - 1) % G + G) % G;
    float r = __ldcg(&vals[((size_t)p * G + src) * T + t]);
    if (r < (float)(p + 1) - 0.5f) atomicAdd(bad, 1u);
    acc += r;
    if (!ok) return;
  }
  if (acc < 0.f) vals[0] = acc;
}

extern "C" __global__ void sync_chain_coop(float* vals, unsigned* bad, int nphase, int work) {
  cg::grid_group grid = cg::this_grid();
  const int G = gridDim.x, b = blockIdx.x, t = threadIdx.x, T = blockDim.x;
  float acc = 0.f;
  for (int p = 0; p < nphase; ++p) {
    float v = (float)(p + 1);
    for (int w = 0; w < work; ++w) v = v * 1.0000001f + 1e-7f;
    vals[((size_t)p * G + b) * T + t] = (p + 1) + (v - v);
    grid.sync();
    float r = __ldcg(&vals[((size_t)p * G + (b + 1) % G) * T + t]);
    if (r < (float)(p + 1) - 0.5f) atomicAdd(bad, 1u);
    acc += r;
  }
  if (acc < 0.f) vals[0] = acc;
}

// stand-in for "one kernel per phase": a trivial kernel, one launch per phase (graph nodes)
extern "C" __global__ void tiny_phase(float* vals, int p, int work) {
  const int G = gridDim.x, b = blockIdx.x, t = threadIdx.x, T = blockDim.x;
  float v = (float)(p + 1);
  for (int w = 0; w < work; ++w) v = v * 1.0000001f + 1e-7f;
  vals[((size_t)p * G + b) * T + t] = (p + 1) + (v - v);
}
"""

SOURCE = COMMON + KERNELS
