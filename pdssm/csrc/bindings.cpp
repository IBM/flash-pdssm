#include <torch/extension.h>
#include "common.h"

// Forward declarations from .cu
//at::Tensor scan_forward(const at::Tensor& P, const at::Tensor& D, const at::Tensor& b);
at::Tensor scan_forward_compact_real(
    const at::Tensor& P_i16,
    const at::Tensor& D_re_bf16,
    const at::Tensor& b_re_bf16);


std::vector<at::Tensor> scan_backward_compact_real(
  const at::Tensor& P_i16,
  const at::Tensor& D_re_bf16,
  const at::Tensor& gb_out_re,
  const at::Tensor& x_re);



// Split entry function for compact-storage forward (mirrors the others)
at::Tensor scan_forward_compact_entry(
  const at::Tensor& P_i16,
  const at::Tensor& D_re_bf16,
  const at::Tensor& b_re_bf16)
{
  CHECK_INPUT(P_i16);
  CHECK_INPUT(D_re_bf16);
  CHECK_INPUT(b_re_bf16);
  // Same shape validation as used in the .cu host function
  check_shapes_compact_real(P_i16, D_re_bf16, b_re_bf16);
  return scan_forward_compact_real(P_i16, D_re_bf16, b_re_bf16);
}


std::vector<at::Tensor> scan_backward_compact_entry(
  const at::Tensor& P_i16,
  const at::Tensor& D_re_bf16,
  const at::Tensor& gb_out_re,
  const at::Tensor& x_re)
{
  CHECK_INPUT(P_i16);
  CHECK_INPUT(D_re_bf16);
  CHECK_INPUT(gb_out_re);
  CHECK_INPUT(x_re);
  // Shape & dtype validation is performed inside the .cu host function.
  return scan_backward_compact_real(P_i16, D_re_bf16, gb_out_re, x_re);
}


at::Tensor scan_forward_strided(const at::Tensor& P_i16, const at::Tensor& D4, const at::Tensor& b4);
std::vector<at::Tensor> scan_backward_strided(
    const at::Tensor& P_i16, const at::Tensor& D4, const at::Tensor& G4, const at::Tensor& X4,
    const at::Tensor& gb_buf);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scan_forward_strided", &scan_forward_strided,
        "Scan on (B,H,L,N) views of any strides; D fp32, b fp32/bf16; returns fp32 x stored as (B,L,H,N)");
  m.def("scan_backward_strided", &scan_backward_strided,
        "Reverse scan on (B,H,L,N) fp32 views of any strides; returns (grad_b, grad_D, D*x_prev)");

  m.def("scan_forward_compact", &scan_forward_compact_entry,
        "Two-phase scan with compact storage (P:int16, D/b: bf16 split) (CUDA)");

m.def("scan_backward_compact", &scan_backward_compact_entry,
        "Reverse scan for grad operator with compact storage (P:int16, D/b: bf16 split) (CUDA)");

}