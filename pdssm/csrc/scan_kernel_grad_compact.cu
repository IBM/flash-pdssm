// Two-phase (block) inclusive scan over time dimension L for triples (P, D, b)
// COLUMN ONE-HOT BACKWARD PASS
// Shapes: P/D/b are (B, L, N). P:int16, D/b: floats.
//
// OPTIMIZATION NOTE (vs baseline):
//   Kernel C previously re-read gb_out_re, D_in, P_in and x_re_in from global
//   memory 2-3x per timestep. On this bandwidth-bound kernel that is the main
//   cost. Each operand is now loaded ONCE per step into a register and reused
//   for all of: v_t, Dy = D_t*x_{t-1}, gD, and the compose staging. Barriers
//   and math are unchanged; only redundant HBM loads are removed.

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "common.h"

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

template <typename scalar_t>
__device__ __forceinline__
float load_real_generic(const scalar_t* __restrict__ src, long long idx) {
  return static_cast<float>(src[idx]);
}

template <typename scalar_t>
__device__ __forceinline__
void store_real_to(float v, scalar_t* __restrict__ dst, long long idx) {
  dst[idx] = scalar_t(v);
}


// -----------------------------------------------------------------------------
// COLUMN ONE-HOT Backward Composition (Pure Gathers, No Atomics)
// Operator (left ⊗ right) where left is chronologically later.
template <typename scalar_t>
__device__ __forceinline__
void compose_inplace_for_thread_grad(
    int tx, int N, bool active,
    int32_t& pi,
    scalar_t& di,
    scalar_t& vi_k,
    const int32_t* __restrict__ sPj,
    const scalar_t* __restrict__ sDj,
    const scalar_t* __restrict__ svj,
    int32_t* __restrict__ sAccP,
    scalar_t* __restrict__ sAccD,
    scalar_t* __restrict__ sAccV)
{
    // Stage the thread-local prefix accumulators into shared memory
    if (active) {
        sAccP[tx] = pi;
        sAccD[tx] = di;
        sAccV[tx] = vi_k;
    }
    __syncthreads();

    // Perform the adjoint composition via purely coalesced gathers
    if (active) {
        int32_t dest = sPj[tx];            // P_j[n]

        pi   = sAccP[dest];                // P_new[n] = P_i[ P_j[n] ]
        di   = sDj[tx] * sAccD[dest];      // D_new[n] = D_j[n] * D_i[ P_j[n] ]
        vi_k = svj[tx] + sDj[tx] * sAccV[dest]; // v_new[n] = v_j[n] + D_j[n] * v_i[ P_j[n] ]
    }
    __syncthreads();
}

// ===== Kernel A (compact storage): per-tile aggregates ======================
template <typename scalar_t>
__global__ void tp_tile_aggregate_kernel_compact_grad(
    const int16_t* __restrict__ P_in,
    const scalar_t* __restrict__ D_in, SeqLayout layD,
    const scalar_t* __restrict__ gb_out_re, SeqLayout layG,
    int32_t* __restrict__ P_tileAgg,
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
  int32_t* sPj   = reinterpret_cast<int32_t*>(smem_raw);
  float*   sDj   = reinterpret_cast<float*>(sPj + N);
  float*   sbj   = sDj + N;
  int32_t* sAccP = reinterpret_cast<int32_t*>(sbj + N);
  float*   sAccD = reinterpret_cast<float*>(sAccP + N);
  float*   sAccV = sAccD + N;

  int32_t pi   = tx;
  float   di   = 1.f;
  float   bi_k = 0.f;

  const int l0  = t * tileL;
  const int len = max(0, min(tileL, L - l0));

  // software pipelining: load step s-1's operands while step s is composed
  int32_t nP = 0; float nD = 1.f, nG = 0.f;
  if (active && len > 0) {
    const int l = l0 + len - 1;
    nP = static_cast<int32_t>(P_in[base_idx(b, L, N, l) + tx]);
    nD = static_cast<float>(D_in[layD.at(b, l, tx)]);
    nG = static_cast<float>(gb_out_re[layG.at(b, l, tx)]);
  }
  for (int s = len - 1; s >= 0; --s) { // run recurrence reversed
    if (active) {
      sPj[tx] = nP; sDj[tx] = nD;
      sAccV[tx] = nG;                                   // G_t[n]
      if (s > 0) {
        const int lp = l0 + s - 1;
        nP = static_cast<int32_t>(P_in[base_idx(b, L, N, lp) + tx]);
        nD = static_cast<float>(D_in[layD.at(b, lp, tx)]);
        nG = static_cast<float>(gb_out_re[layG.at(b, lp, tx)]);
      }
    }
    __syncthreads();
    if (active) {
      int32_t dest = sPj[tx];
      sbj[tx] = sDj[tx] * sAccV[dest];
    }
    __syncthreads();
    compose_inplace_for_thread_grad<float>(
        tx, N, active, pi, di, bi_k, sPj, sDj, sbj, sAccP, sAccD, sAccV);
  }

  const long long base_t = tile_idx(b, numTiles, N, t);
  if (active) {
    P_tileAgg[base_t + tx] = pi;
    D_tileAgg[base_t + tx] = di;
    b_tileAgg[base_t + tx] = bi_k;
  }
}

__global__ void tp_prefix_scan_tiles_kernel_compact_grad(
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
  int32_t* sPj   = reinterpret_cast<int32_t*>(smem_raw);
  float*   sDj   = reinterpret_cast<float*>(sPj + N);
  float*   sbj   = sDj + N;
  int32_t* sAccP = reinterpret_cast<int32_t*>(sbj + N);
  float*   sAccD = reinterpret_cast<float*>(sAccP + N);
  float*   sAccV = sAccD + N;

  int32_t pi   = tx;
  float   di   = 1.f;
  float   bi_k = 0.f;

  int32_t nP = 0; float nD = 1.f, nB = 0.f;
  if (active && numTiles > 0) {
    const long long bl = tile_idx(b, numTiles, N, numTiles - 1);
    nP = P_tileAgg[bl + tx]; nD = D_tileAgg[bl + tx]; nB = b_tileAgg[bl + tx];
  }
  for (int t = numTiles - 1; t >= 0; --t) {
    long long base_t = tile_idx(b, numTiles, N, t);
    if (active) {
      P_prefix[base_t + tx] = pi;
      D_prefix[base_t + tx] = di;
      b_prefix[base_t + tx] = bi_k;
      sPj[tx] = nP; sDj[tx] = nD; sbj[tx] = nB;
      if (t > 0) {
        const long long bp = tile_idx(b, numTiles, N, t - 1);
        nP = P_tileAgg[bp + tx]; nD = D_tileAgg[bp + tx]; nB = b_tileAgg[bp + tx];
      }
    }
    __syncthreads();
    compose_inplace_for_thread_grad<float>(
        tx, N, active, pi, di, bi_k, sPj, sDj, sbj, sAccP, sAccD, sAccV);
  }
}

// Writes, for every (seq, l, n):
//   gb[l]  = v_l = G_l + (carried reverse-scan state)        (dL/db_l = dL/dx_l, "lambda")
//   gD[l]  = v_l[P_l[n]] * x_{l-1}[n]                         (dL/dD_l; 0 at l = 0)
//   Dy[l]  = D_l[n] * x_{l-1}[n]                              (input of the dictionary/selector grads)
template <typename scalar_t>
__global__ void tp_apply_prefix_kernel_compact_grad(
    const int16_t* __restrict__ P_in,
    const scalar_t* __restrict__ D_in, SeqLayout layD,
    const int32_t* __restrict__ P_prefix,
    const float*   __restrict__ D_prefix,
    const float*   __restrict__ b_prefix,
    const scalar_t* __restrict__ gb_out_re, SeqLayout layG,
    const scalar_t* __restrict__ x_re_in, SeqLayout layX,
    float* __restrict__ gD_re_out, SeqLayout layGD,
    scalar_t* __restrict__ D_re_out, SeqLayout layDy,
    scalar_t* __restrict__ gb_re, SeqLayout layGB,
    int B, int L, int N, int tileL, int numTiles)
{
  const int b = blockIdx.z;
  const int t = blockIdx.y;
  const int tx = threadIdx.x;
  const bool active = (tx < N);
  if (b >= B || t >= numTiles) return;

  extern __shared__ unsigned char smem_raw[];
  int32_t* sPj   = reinterpret_cast<int32_t*>(smem_raw);
  float*   sDj   = reinterpret_cast<float*>(sPj + N);
  float*   sbj   = sDj + N;
  int32_t* sAccP = reinterpret_cast<int32_t*>(sbj + N);
  float*   sAccD = reinterpret_cast<float*>(sAccP + N);
  float*   sAccV = sAccD + N;

  const long long base_t = tile_idx(b, numTiles, N, t);
  int32_t pi   = active ? P_prefix[base_t + tx] : 0;
  float   di   = active ? D_prefix[base_t + tx] : 1.f;
  float   bi_k = active ? b_prefix[base_t + tx] : 0.f;

  const int l0  = t * tileL;
  const int len = max(0, min(tileL, L - l0));

  // software pipelining: operands of step l-1 are loaded while step l is processed
  float nG = 0.f, nDd = 0.f, nX = 0.f; int32_t nPp = 0;
  auto load_step = [&](int l) {
    nG  = static_cast<float>(gb_out_re[layG.at(b, l, tx)]);
    nDd = static_cast<float>(D_in[layD.at(b, l, tx)]);
    nPp = static_cast<int32_t>(P_in[base_idx(b, L, N, l) + tx]);
    nX  = (l > 0) ? static_cast<float>(x_re_in[layX.at(b, l - 1, tx)]) : 0.f;
  };
  if (active && len > 0) load_step(l0 + len - 1);

  for (int s = len - 1; s >= 0; --s) {
    const int l = l0 + s;

    float   g_t   = nG;    // G_t[n]  = dL/dy_t[n]  (incoming grad)
    float   d_t   = nDd;   // D_t[n]
    float   xprev = nX;    // x_{t-1}[n]  (0 at l = 0)
    int32_t p_t   = nPp;   // P_t[n]
    if (active && s > 0) load_step(l - 1);

    float out_r = 0.f;
    if (active) {
      out_r = bi_k + g_t;              // v_t[n]
      gb_re[layGB.at(b, l, tx)] = scalar_t(out_r);
      sAccV[tx] = out_r;               // stage v_t for the gD gather below
      D_re_out[layDy.at(b, l, tx)] = scalar_t(d_t * xprev);  // xprev = 0 when l == 0
    }
    __syncthreads();
    if (active) {
      gD_re_out[layGD.at(b, l, tx)] = sAccV[p_t] * xprev;     // 0 when l == 0
    }
    __syncthreads();

    if (active) {
      sPj[tx] = p_t;
      sDj[tx] = d_t;
      sAccV[tx] = g_t;                 // G_t[n]
    }
    __syncthreads();
    if (active) {
      sbj[tx] = d_t * sAccV[p_t];      // == sDj[tx] * sAccV[sPj[tx]]
    }
    __syncthreads();
    compose_inplace_for_thread_grad<float>(
        tx, N, active, pi, di, bi_k, sPj, sDj, sbj, sAccP, sAccD, sAccV);
  }
}

// --------------------------- shared host driver ------------------------------
template <typename scalar_t>
static void launch_backward(
    const at::Tensor& P_i16, const scalar_t* D, SeqLayout layD, const scalar_t* G, SeqLayout layG,
    const scalar_t* X, SeqLayout layX, float* gD, SeqLayout layGD, scalar_t* Dy, SeqLayout layDy,
    scalar_t* gb, SeqLayout layGB, int B, int L, int N)
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
  const size_t shmem_bytes = (2 * sizeof(int32_t) + 4 * sizeof(float)) * (size_t)N;
  auto stream = at::cuda::getCurrentCUDAStream();

  tp_tile_aggregate_kernel_compact_grad<scalar_t><<<dim3(1, numTiles, B), threads, shmem_bytes, stream>>>(
      P_i16.data_ptr<int16_t>(), D, layD, G, layG,
      P_tileAgg.data_ptr<int32_t>(), D_tileAgg.data_ptr<float>(), b_tileAgg.data_ptr<float>(),
      B, L, N, tileL, numTiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  tp_prefix_scan_tiles_kernel_compact_grad<<<dim3(1, 1, B), threads, shmem_bytes, stream>>>(
      P_tileAgg.data_ptr<int32_t>(), D_tileAgg.data_ptr<float>(), b_tileAgg.data_ptr<float>(),
      P_prefix.data_ptr<int32_t>(), D_prefix.data_ptr<float>(), b_prefix.data_ptr<float>(),
      B, N, numTiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  tp_apply_prefix_kernel_compact_grad<scalar_t><<<dim3(1, numTiles, B), threads, shmem_bytes, stream>>>(
      P_i16.data_ptr<int16_t>(), D, layD,
      P_prefix.data_ptr<int32_t>(), D_prefix.data_ptr<float>(), b_prefix.data_ptr<float>(),
      G, layG, X, layX, gD, layGD, Dy, layDy, gb, layGB, B, L, N, tileL, numTiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ==== host entry: contiguous (num_seq, L, N) operands (original API) =============
std::vector<at::Tensor> scan_backward_compact_real(
    const at::Tensor& P_i16,
    const at::Tensor& D_re_in,
    const at::Tensor& gb_out_re,
    const at::Tensor& x_re)
{
  check_shapes_compact_real(P_i16, D_re_in, gb_out_re);
  const int B = (int)P_i16.size(0);
  const int L = (int)P_i16.size(1);
  const int N = (int)P_i16.size(2);
  TORCH_CHECK(D_re_in.scalar_type() == gb_out_re.scalar_type() && D_re_in.scalar_type() == x_re.scalar_type(),
              "D, grad and x must share a dtype");
  at::Tensor gb_re = at::empty_like(gb_out_re);
  at::Tensor Dy_re = at::empty_like(D_re_in);
  at::Tensor gD_re = at::empty({B, L, N}, D_re_in.options().dtype(at::kFloat));
  const SeqLayout lay = layout_contig(L, N);
  if (D_re_in.scalar_type() == at::kBFloat16) {
    launch_backward<at::BFloat16>(
        P_i16, D_re_in.data_ptr<at::BFloat16>(), lay, gb_out_re.data_ptr<at::BFloat16>(), lay,
        x_re.data_ptr<at::BFloat16>(), lay, gD_re.data_ptr<float>(), lay,
        Dy_re.data_ptr<at::BFloat16>(), lay, gb_re.data_ptr<at::BFloat16>(), lay, B, L, N);
  } else {
    TORCH_CHECK(D_re_in.scalar_type() == at::kFloat, "D/b must be bf16 or float32");
    launch_backward<float>(
        P_i16, D_re_in.data_ptr<float>(), lay, gb_out_re.data_ptr<float>(), lay,
        x_re.data_ptr<float>(), lay, gD_re.data_ptr<float>(), lay,
        Dy_re.data_ptr<float>(), lay, gb_re.data_ptr<float>(), lay, B, L, N);
  }
  return {gb_re, gD_re, Dy_re};
}

// empty tensor with the same (dense) strides as `like`, else contiguous
static at::Tensor empty_like_strided(const at::Tensor& like, at::ScalarType dtype) {
  if (like.is_non_overlapping_and_dense()) {
    return at::empty_strided(like.sizes(), like.strides(), like.options().dtype(dtype));
  }
  return at::empty(like.sizes(), like.options().dtype(dtype));
}

// ==== host entry: strided (B, H, L, N) views, all fp32 ===========================
// D (fp32), grad_x and x: (B, H, L, N), any strides. gb_buf: fp32 (B, H, L, N) output buffer for
// dL/db, typically allocated by the caller with b's strides so the gradient flows back through
// the caller's permute/view of b without a copy. Returns
//   gb  dL/db  (= gb_buf)
//   gD  dL/dD  with D's strides
//   Dy  D_t * x_{t-1}, laid out like x
std::vector<at::Tensor> scan_backward_strided(
    const at::Tensor& P_i16, const at::Tensor& D4, const at::Tensor& G4, const at::Tensor& X4,
    const at::Tensor& gb_buf)
{
  CHECK_INPUT(P_i16);
  CHECK_CUDA(D4); CHECK_CUDA(G4); CHECK_CUDA(X4); CHECK_CUDA(gb_buf);
  TORCH_CHECK(D4.dim() == 4 && D4.sizes() == G4.sizes() && D4.sizes() == X4.sizes() && D4.sizes() == gb_buf.sizes(),
              "D, grad_x, x, grad_b must all be (B, H, L, N)");
  TORCH_CHECK(gb_buf.scalar_type() == at::kFloat, "grad_b buffer must be float32");
  TORCH_CHECK(D4.scalar_type() == at::kFloat && G4.scalar_type() == at::kFloat && X4.scalar_type() == at::kFloat,
              "D, grad_x and x must be float32");
  const int Bb = (int)D4.size(0), H = (int)D4.size(1), L = (int)D4.size(2), N = (int)D4.size(3);
  TORCH_CHECK(P_i16.size(0) == (long long)Bb * H && P_i16.size(1) == L && P_i16.size(2) == N, "P must be (B*H, L, N)");
  at::Tensor gb = gb_buf;
  at::Tensor gD = empty_like_strided(D4, at::kFloat);
  at::Tensor Dy = empty_like_strided(X4, at::kFloat);
  launch_backward<float>(
      P_i16, D4.data_ptr<float>(), layout_of(D4), G4.data_ptr<float>(), layout_of(G4),
      X4.data_ptr<float>(), layout_of(X4), gD.data_ptr<float>(), layout_of(gD),
      Dy.data_ptr<float>(), layout_of(Dy), gb.data_ptr<float>(), layout_of(gb), Bb * H, L, N);
  return {gb, gD, Dy};
}
