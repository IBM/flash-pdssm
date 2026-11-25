// Two-phase (block) inclusive scan over time dimension L for triples (P, D, b)
// Shapes: P/D/b are (B, L, N). P:int16, D/b: floats.
// Operator (left ⊗ right):
//   P_new[n] = P_right[ P_left[n] ]
//   D_new[n] = D_right[ P_left[n] ] * D_left[n]
//   COLUMN ONE-HOT b-rule (scatter-add):
//     b_new[m] = b_right[m] + sum_{n: P_right[n]=m} ( D_right[n] * b_left[n] )
//
// OPTIMIZATION NOTE (vs baseline):
//   The shared-memory accumulator sAcc used by the b-rule scatter is now zeroed
//   in each kernel's LOAD phase (before the load barrier) instead of inside
//   compose_inplace_for_thread. That makes the pre-scatter barrier that used to
//   sit between "zero sAcc" and "scatter" redundant, dropping one __syncthreads()
//   per timestep (4 -> 3 barriers/step in the tiled loop). Math is unchanged.

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "common.h"

// Tile length over the time axis. The paper reports tau=128 as optimal; 256 was
// the baseline here. This is a cheap knob to sweep for your shapes.
#ifndef PDSSM_TILE_L
#define PDSSM_TILE_L 256
#endif

static inline int div_ceil_int(int a, int b) { return (a + b - 1) / b; }

__host__ __device__ inline long long base_idx(int b, int L, int N, int l) {
  return (static_cast<long long>(b) * L + l) * N;
}
__host__ __device__ inline long long tile_idx(int b, int numTiles, int N, int t) {
  return (static_cast<long long>(b) * numTiles + t) * N;
}

template <typename T>
__device__ __forceinline__
float load_real_from(const T* __restrict__ x, long long idx) {
  return static_cast<float>(x[idx]);
}

template <typename T>
__device__ __forceinline__
void store_real_to(float v, T* __restrict__ out, long long idx) {
  out[idx] = T(v);
}

// -------------------- device compose (one step) --------------------
// Implements binary operator: left ⊗ right, with right staged in shared memory
// (sPj,sDj,sbj) and shared accumulator sAcc for the scatter-add part of b.
//
// PRECONDITION: sAcc[0..N) has been zeroed by the caller in the load phase, and
// the caller's post-load __syncthreads() has made those zeros visible to all
// threads. compose therefore no longer zeroes sAcc and no longer needs its own
// pre-scatter barrier.
template <typename real_t>
__device__ __forceinline__
void compose_inplace_for_thread(
    int tx, int N, bool active,
    int32_t& pi, real_t& di, real_t& bi_k,
    const int32_t* __restrict__ sPj,
    const real_t* __restrict__ sDj,
    const real_t* __restrict__ sbj,
    real_t* __restrict__ sAcc)
{
  int32_t newP = 0;
  real_t newD = real_t(1);

  if (active) {
    // newP[n] = sPj[pi_old]; newD[n] = sDj[pi_old] * di_old
    const int32_t pi_old = pi;
    newP = sPj[pi_old];
    newD = sDj[pi_old] * di;

    // COLUMN ONE-HOT b-rule (scatter-add): each source node pushes its scaled
    // historical bias forward. sAcc was pre-zeroed by the caller.
    const real_t val_to_scatter = sDj[tx] * bi_k;
    const int32_t dest = sPj[tx];
    atomicAdd(&sAcc[dest], val_to_scatter);
  }
  __syncthreads(); // Ensure all scatters are completely resolved

  if (active) {
    // Read the aggregated collisions destined for this lane and add to the new input
    bi_k = sbj[tx] + sAcc[tx];

    pi = newP;
    di = newD;
  }
  __syncthreads();
}

// ===== Kernel A: per-tile aggregates ========================================
// P is (num_seq, L, N) contiguous int16; D, b are read through their layouts.
template <typename TD, typename TB>
__global__ void tp_tile_aggregate_kernel_compact(
    const int16_t* __restrict__ P_in,
    const TD* __restrict__ D_in, SeqLayout layD,
    const TB* __restrict__ b_in, SeqLayout layB,
    int32_t* __restrict__ P_tileAgg,    // [num_seq,numTiles,N]
    float* __restrict__ D_tileAgg,
    float* __restrict__ b_tileAgg,
    int B, int L, int N, int tileL, int numTiles)
{
  const int b = blockIdx.z;
  const int t = blockIdx.y;
  const int tx = threadIdx.x;
  if (b >= B || t >= numTiles) return;
  const bool active = (tx < N);

  extern __shared__ unsigned char smem_raw[];
  int32_t* sPj = reinterpret_cast<int32_t*>(smem_raw);
  float*   sDj = reinterpret_cast<float*>(sPj + N);
  float*   sbj = sDj + N;
  float*   sAcc = sbj + N;

  int32_t pi = tx;
  float   di = 1.f;
  float   bi_k = 0.f;

  const int l0  = t * tileL;
  const int len = max(0, min(tileL, L - l0));

  // software pipelining: step s+1's operands are loaded into registers while step s is
  // composed, so the global-load latency overlaps the barriers instead of preceding them
  int32_t nP = 0; float nD = 1.f, nB = 0.f;
  if (active && len > 0) {
    nP = static_cast<int32_t>(P_in[base_idx(b, L, N, l0) + tx]);
    nD = static_cast<float>(D_in[layD.at(b, l0, tx)]);
    nB = static_cast<float>(b_in[layB.at(b, l0, tx)]);
  }
  for (int s = 0; s < len; ++s) {
    if (active) {
      sPj[tx] = nP; sDj[tx] = nD; sbj[tx] = nB;
      sAcc[tx] = 0.f;   // zero scatter accumulator here (folds one barrier)
      if (s + 1 < len) {
        const int ln = l0 + s + 1;
        nP = static_cast<int32_t>(P_in[base_idx(b, L, N, ln) + tx]);
        nD = static_cast<float>(D_in[layD.at(b, ln, tx)]);
        nB = static_cast<float>(b_in[layB.at(b, ln, tx)]);
      }
    }
    __syncthreads();

    compose_inplace_for_thread<float>(
        tx, N, active, pi, di, bi_k, sPj, sDj, sbj, sAcc);
  }

  const long long base_t = tile_idx(b, numTiles, N, t);
  if (active) {
    P_tileAgg[base_t + tx] = pi;
    D_tileAgg[base_t + tx] = di;
    b_tileAgg[base_t + tx] = bi_k;
  }
}

// ===== Kernel B: scan tile aggregates (exclusive) ===========================
__global__ void tp_prefix_scan_tiles_kernel_compact(
    const int32_t* __restrict__ P_tileAgg,
    const float*   __restrict__ D_tileAgg,
    const float*   __restrict__ b_tileAgg,
    int32_t* __restrict__ P_prefix,
    float*   __restrict__ D_prefix,
    float*   __restrict__ b_prefix,
    int B, int N, int numTiles)
{
  const int b  = blockIdx.z;
  const int tx = threadIdx.x;
  const bool active = (tx < N);
  if (b >= B) return;

  extern __shared__ unsigned char smem_raw[];
  int32_t* sPj = reinterpret_cast<int32_t*>(smem_raw);
  float*   sDj = reinterpret_cast<float*>(sPj + N);
  float*   sbj = sDj + N;
  float*   sAcc = sbj + N;

  int32_t pi = tx;
  float   di = 1.f;
  float   bi_k = 0.f;

  int32_t nP = 0; float nD = 1.f, nB = 0.f;
  if (active && numTiles > 0) {
    const long long b0 = tile_idx(b, numTiles, N, 0);
    nP = P_tileAgg[b0 + tx]; nD = D_tileAgg[b0 + tx]; nB = b_tileAgg[b0 + tx];
  }
  for (int t = 0; t < numTiles; ++t) {
    const long long base_t = tile_idx(b, numTiles, N, t);
    if (active) {
      P_prefix[base_t + tx] = pi;
      D_prefix[base_t + tx] = di;
      b_prefix[base_t + tx] = bi_k;

      sPj[tx] = nP; sDj[tx] = nD; sbj[tx] = nB;
      sAcc[tx] = 0.f;
      if (t + 1 < numTiles) {
        const long long bn = tile_idx(b, numTiles, N, t + 1);
        nP = P_tileAgg[bn + tx]; nD = D_tileAgg[bn + tx]; nB = b_tileAgg[bn + tx];
      }
    }
    __syncthreads();

    compose_inplace_for_thread<float>(
        tx, N, active, pi, di, bi_k, sPj, sDj, sbj, sAcc);
  }
}

// ===== Kernel C: apply prefix -> outputs =====================================
template <typename TD, typename TB, typename TO>
__global__ void tp_apply_prefix_kernel_compact(
    const int16_t* __restrict__ P_in,
    const TD* __restrict__ D_in, SeqLayout layD,
    const TB* __restrict__ b_in, SeqLayout layB,
    const int32_t* __restrict__ P_prefix,
    const float*   __restrict__ D_prefix,
    const float*   __restrict__ b_prefix,
    TO* __restrict__ b_out, SeqLayout layO,
    int B, int L, int N, int tileL, int numTiles)
{
  const int b = blockIdx.z;
  const int t = blockIdx.y;
  const int tx = threadIdx.x;
  const bool active = (tx < N);
  if (b >= B || t >= numTiles) return;

  extern __shared__ unsigned char smem_raw[];
  int32_t* sPj = reinterpret_cast<int32_t*>(smem_raw);
  float*   sDj = reinterpret_cast<float*>(sPj + N);
  float*   sbj = sDj + N;
  float*   sAcc = sbj + N;

  const long long base_t = tile_idx(b, numTiles, N, t);
  int32_t pi  = active ? P_prefix[base_t + tx] : 0;
  float   di  = active ? D_prefix[base_t + tx] : 1.f;
  float   bi_k = active ? b_prefix[base_t + tx] : 0.f;

  const int l0  = t * tileL;
  const int len = max(0, min(tileL, L - l0));

  int32_t nP = 0; float nD = 1.f, nB = 0.f;
  if (active && len > 0) {
    nP = static_cast<int32_t>(P_in[base_idx(b, L, N, l0) + tx]);
    nD = static_cast<float>(D_in[layD.at(b, l0, tx)]);
    nB = static_cast<float>(b_in[layB.at(b, l0, tx)]);
  }
  for (int s = 0; s < len; ++s) {
    const int l = l0 + s;
    if (active) {
      sPj[tx] = nP; sDj[tx] = nD; sbj[tx] = nB;
      sAcc[tx] = 0.f;
      if (s + 1 < len) {
        nP = static_cast<int32_t>(P_in[base_idx(b, L, N, l + 1) + tx]);
        nD = static_cast<float>(D_in[layD.at(b, l + 1, tx)]);
        nB = static_cast<float>(b_in[layB.at(b, l + 1, tx)]);
      }
    }
    __syncthreads();

    compose_inplace_for_thread<float>(
        tx, N, active, pi, di, bi_k, sPj, sDj, sbj, sAcc);

    // compose ends with a barrier, so the next iteration may overwrite shared memory
    if (active) {
      b_out[layO.at(b, l, tx)] = TO(bi_k);
    }
  }
}

// --------------------------- shared host driver ------------------------------
template <typename TD, typename TB, typename TO>
static void launch_forward(
    const at::Tensor& P_i16, const TD* D, SeqLayout layD, const TB* bb, SeqLayout layB,
    TO* out, SeqLayout layO, int B, int L, int N)
{
  auto round_up_warp = [](int x) { return ((x + 31) / 32) * 32; };
  TORCH_CHECK(N <= 1024, "the scan kernel requires N <= 1024 (got N=", N, ")");
  const int threads = round_up_warp(N);
  const int tileL   = pdssm_tile_len(PDSSM_TILE_L, B, L);  // B = number of sequences (batch * heads)
  const int numTiles = div_ceil_int(L, tileL);

  at::TensorOptions optI32 = P_i16.options().dtype(at::kInt);
  at::TensorOptions optF32 = P_i16.options().dtype(at::kFloat);
  at::Tensor P_tileAgg = at::empty({B, numTiles, N}, optI32);
  at::Tensor D_tileAgg = at::empty({B, numTiles, N}, optF32);
  at::Tensor b_tileAgg = at::empty({B, numTiles, N}, optF32);
  at::Tensor P_prefix  = at::empty({B, numTiles, N}, optI32);
  at::Tensor D_prefix  = at::empty({B, numTiles, N}, optF32);
  at::Tensor b_prefix  = at::empty({B, numTiles, N}, optF32);
  const size_t shmem_bytes = (sizeof(int32_t) + 3 * sizeof(float)) * (size_t)N;
  auto stream = at::cuda::getCurrentCUDAStream();

  tp_tile_aggregate_kernel_compact<TD, TB><<<dim3(1, numTiles, B), threads, shmem_bytes, stream>>>(
      P_i16.data_ptr<int16_t>(), D, layD, bb, layB,
      P_tileAgg.data_ptr<int32_t>(), D_tileAgg.data_ptr<float>(), b_tileAgg.data_ptr<float>(),
      B, L, N, tileL, numTiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  tp_prefix_scan_tiles_kernel_compact<<<dim3(1, 1, B), threads, shmem_bytes, stream>>>(
      P_tileAgg.data_ptr<int32_t>(), D_tileAgg.data_ptr<float>(), b_tileAgg.data_ptr<float>(),
      P_prefix.data_ptr<int32_t>(), D_prefix.data_ptr<float>(), b_prefix.data_ptr<float>(),
      B, N, numTiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  tp_apply_prefix_kernel_compact<TD, TB, TO><<<dim3(1, numTiles, B), threads, shmem_bytes, stream>>>(
      P_i16.data_ptr<int16_t>(), D, layD, bb, layB,
      P_prefix.data_ptr<int32_t>(), D_prefix.data_ptr<float>(), b_prefix.data_ptr<float>(),
      out, layO, B, L, N, tileL, numTiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ==== host entry: contiguous (num_seq, L, N) operands, D/b/out share a dtype ====
at::Tensor scan_forward_compact_real(
    const at::Tensor& P_i16,
    const at::Tensor& D_bf16,
    const at::Tensor& b_bf16)
{
  check_shapes_compact_real(P_i16, D_bf16, b_bf16);
  const int B = (int)P_i16.size(0);
  const int L = (int)P_i16.size(1);
  const int N = (int)P_i16.size(2);
  TORCH_CHECK(D_bf16.scalar_type() == b_bf16.scalar_type(), "D and b must share a dtype");
  at::Tensor b_out = at::empty_like(b_bf16);
  const SeqLayout lay = layout_contig(L, N);
  if (D_bf16.scalar_type() == at::kBFloat16) {
    launch_forward<at::BFloat16, at::BFloat16, at::BFloat16>(
        P_i16, D_bf16.data_ptr<at::BFloat16>(), lay, b_bf16.data_ptr<at::BFloat16>(), lay,
        b_out.data_ptr<at::BFloat16>(), lay, B, L, N);
  } else {
    TORCH_CHECK(D_bf16.scalar_type() == at::kFloat, "D/b tensors must be bf16 or float32");
    launch_forward<float, float, float>(
        P_i16, D_bf16.data_ptr<float>(), lay, b_bf16.data_ptr<float>(), lay,
        b_out.data_ptr<float>(), lay, B, L, N);
  }
  return b_out;
}

// ==== host entry: strided (B, H, L, N) views, no copies =========================
// P: (B*H, L, N) contiguous int16. D: fp32, any strides. b: fp32 or bf16, any strides.
// Returns x as an fp32 (B, H, L, N) view of (B, L, H, N) storage, so x.transpose(1, 2) is a
// contiguous (B, L, H, N) tensor (the layout the output norm / projection consume).
at::Tensor scan_forward_strided(const at::Tensor& P_i16, const at::Tensor& D4, const at::Tensor& b4)
{
  CHECK_INPUT(P_i16);
  CHECK_CUDA(D4); CHECK_CUDA(b4);
  TORCH_CHECK(P_i16.scalar_type() == at::kShort, "P must be int16");
  TORCH_CHECK(D4.dim() == 4 && b4.dim() == 4 && D4.sizes() == b4.sizes(), "D, b must be (B, H, L, N)");
  const int Bb = (int)D4.size(0), H = (int)D4.size(1), L = (int)D4.size(2), N = (int)D4.size(3);
  TORCH_CHECK(P_i16.size(0) == (long long)Bb * H && P_i16.size(1) == L && P_i16.size(2) == N, "P must be (B*H, L, N)");
  TORCH_CHECK(D4.scalar_type() == at::kFloat, "D must be float32");
  at::Tensor out = at::empty({Bb, L, H, N}, D4.options().dtype(at::kFloat)).transpose(1, 2);
  const SeqLayout layD = layout_of(D4), layB = layout_of(b4), layO = layout_of(out);
  if (b4.scalar_type() == at::kBFloat16) {
    launch_forward<float, at::BFloat16, float>(
        P_i16, D4.data_ptr<float>(), layD, b4.data_ptr<at::BFloat16>(), layB,
        out.data_ptr<float>(), layO, Bb * H, L, N);
  } else {
    TORCH_CHECK(b4.scalar_type() == at::kFloat, "b must be float32 or bfloat16");
    launch_forward<float, float, float>(
        P_i16, D4.data_ptr<float>(), layD, b4.data_ptr<float>(), layB,
        out.data_ptr<float>(), layO, Bb * H, L, N);
  }
  return out;
}
