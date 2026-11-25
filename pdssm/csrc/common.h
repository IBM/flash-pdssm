#pragma once
//#include <torch/extension.h>
#include <ATen/ATen.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#include <cstdlib>

// Where element (seq, l, n) of a (num_seq, L, N) operand lives, seq = b * H + h. Lets the scan
// read and write (B, H, L, N) views of any strides (e.g. (B, L, H, N) storage) without copies.
struct SeqLayout {
  int H;
  long long sB, sH, sL, sN;
#ifdef __CUDACC__
  __host__ __device__ __forceinline__
#endif
  long long at(int seq, int l, int n) const {
    const int bb = seq / H;
    const int hh = seq - bb * H;
    return (long long)bb * sB + (long long)hh * sH + (long long)l * sL + (long long)n * sN;
  }
};

// (num_seq, L, N) contiguous
inline SeqLayout layout_contig(int L, int N) {
  return SeqLayout{1, (long long)L * N, 0, (long long)N, 1};
}
// a 4-D (B, H, L, N) tensor with arbitrary strides
inline SeqLayout layout_of(const at::Tensor& t) {
  TORCH_CHECK(t.dim() == 4, "expected a (B, H, L, N) tensor");
  return SeqLayout{(int)t.size(1), t.stride(0), t.stride(1), t.stride(2), t.stride(3)};
}

// Time-tile length of the two-phase scan, chosen at run time.
//   PDSSM_TILE_L=<n>  use n (e.g. to sweep 32/64/128/256 without rebuilding)
//   unset             automatic: about 8192 tiles in total (num_seq * L / tile), as a power of
//                     two in [32, max_tile]. Measured on H200 at L=2048, N=38 (pipelined
//                     kernels) this is within 1% of the fastest tile from 100M to 3B.
// Each tile is processed sequentially by one block, and the tile prefixes by one block per
// sequence, so the serial chain is ~2*tileL + L/tileL steps: shorter tiles mean more blocks
// and a shorter chain, until there are enough blocks to fill the GPU.
inline int pdssm_tile_len(int max_tile, long long num_seq, int L) {
  const char* s = std::getenv("PDSSM_TILE_L");
  if (s != nullptr) {
    int v = std::atoi(s);
    if (v > 0) return v;
  }
  const long long target_tiles = 8192;
  long long want = (num_seq * (long long)L + target_tiles - 1) / target_tiles;
  int tile = 32;
  while (tile < want && tile < max_tile) tile *= 2;
  return tile;
}


// New: checks for compact-storage forward (P:int16, D/b split bf16 planes)
inline void check_shapes_compact_real(
    const at::Tensor& P_i16,
    const at::Tensor& D_re_bf16,
    const at::Tensor& b_re_bf16) {
  CHECK_INPUT(P_i16);
  CHECK_INPUT(D_re_bf16);
  CHECK_INPUT(b_re_bf16);
  TORCH_CHECK(P_i16.dim()==3 && D_re_bf16.dim()==3
              && b_re_bf16.dim()==3, "All inputs must be (B,L,N)");
  TORCH_CHECK(P_i16.sizes() == D_re_bf16.sizes()
              && P_i16.sizes() == b_re_bf16.sizes(), "Shape mismatch");
  TORCH_CHECK(P_i16.scalar_type() == at::kShort, "P must be int16");
}