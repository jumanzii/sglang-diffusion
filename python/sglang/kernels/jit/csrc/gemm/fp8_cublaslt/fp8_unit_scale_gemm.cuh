/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cstdint>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <map>
#include <mutex>
#include <tuple>

namespace sglang {

using namespace host;

namespace fp8_cublaslt_impl {

inline constexpr int kMaxAlgos = 8;

// Descriptors and heuristic results for one (device, M, N, K, workspace) problem.
struct Problem {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t layout_w = nullptr, layout_a = nullptr, layout_d = nullptr;
  cublasLtMatmulHeuristicResult_t algos[kMaxAlgos];
  int num_algos = 0;
};

inline cublasLtHandle_t handle() {
  static cublasLtHandle_t h = [] {
    cublasLtHandle_t x = nullptr;
    CHECK_HOST(cublasLtCreate(&x) == CUBLAS_STATUS_SUCCESS) << "cublasLtCreate failed";
    return x;
  }();
  return h;
}

// Unit per-tensor scales, one per device; cuBLASLt reads them from device memory.
inline const float* device_one(int32_t device) {
  static std::mutex mu;
  static std::map<int32_t, float*> ones;
  std::lock_guard<std::mutex> lock(mu);
  auto it = ones.find(device);
  if (it != ones.end()) return it->second;
  float* p = nullptr;
  const float one = 1.0f;
  CHECK_HOST(cudaMalloc(&p, sizeof(float)) == cudaSuccess) << "cudaMalloc failed";
  CHECK_HOST(cudaMemcpy(p, &one, sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess) << "cudaMemcpy failed";
  return ones.emplace(device, p).first->second;
}

// cuBLASLt is column major, so out[M, N] = a[M, K] @ w[N, K]^T is the TN problem
// D'[N, M] = op(W)[N, K] x A'[K, M] with W as cuBLASLt's "A".
inline const Problem& problem(int32_t device, int64_t m, int64_t n, int64_t k, size_t workspace_bytes) {
  static std::mutex mu;
  static std::map<std::tuple<int32_t, int64_t, int64_t, int64_t, size_t>, Problem> cache;
  std::lock_guard<std::mutex> lock(mu);
  const auto key = std::make_tuple(device, m, n, k, workspace_bytes);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;

  Problem p;
  CHECK_HOST(cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS);
  const cublasOperation_t trans_w = CUBLAS_OP_T, trans_a = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_w, sizeof(trans_w));
  cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_a, sizeof(trans_a));
  const float* one = device_one(device);
  cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &one, sizeof(one));
  cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &one, sizeof(one));
  cublasLtMatrixLayoutCreate(&p.layout_w, CUDA_R_8F_E4M3, k, n, k);
  cublasLtMatrixLayoutCreate(&p.layout_a, CUDA_R_8F_E4M3, k, m, k);
  cublasLtMatrixLayoutCreate(&p.layout_d, CUDA_R_16BF, n, m, n);

  cublasLtMatmulPreference_t pref = nullptr;
  cublasLtMatmulPreferenceCreate(&pref);
  cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes));
  const cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
      handle(), p.desc, p.layout_w, p.layout_a, p.layout_d, p.layout_d, pref, kMaxAlgos, p.algos, &p.num_algos);
  cublasLtMatmulPreferenceDestroy(pref);
  CHECK_HOST(st == CUBLAS_STATUS_SUCCESS && p.num_algos > 0)
      << "no cuBLASLt FP8 algo for M=" << m << " N=" << n << " K=" << k << " (status " << static_cast<int>(st) << ")";
  return cache.emplace(key, p).first->second;
}

}  // namespace fp8_cublaslt_impl

/**
 * \brief Number of cuBLASLt heuristic algos (at most 8) for out[M, N] = a[M, K] @ w[N, K]^T.
 * \param a [M, K] e4m3, row major.
 * \param w [N, K] e4m3, row major.
 * \param workspace uint8 buffer the GEMM may use; its size is part of the heuristic query.
 */
inline int64_t
fp8_unit_scale_gemm_num_algos(tvm::ffi::TensorView a, tvm::ffi::TensorView w, tvm::ffi::TensorView workspace) {
  return fp8_cublaslt_impl::problem(
             a.device().device_id, a.size(0), w.size(0), a.size(1), static_cast<size_t>(workspace.numel()))
      .num_algos;
}

/**
 * \brief out[M, N] (bf16) = a[M, K] @ w[N, K]^T with e4m3 inputs, fp32 accumulation and unit
 * scales, through cuBLASLt's algo_index-th heuristic. The caller applies any scales.
 * \param out [M, N] bf16, row major.
 * \param a [M, K] e4m3, row major.
 * \param w [N, K] e4m3, row major (the stored linear weight).
 * \param workspace uint8 buffer for cuBLASLt.
 * \param algo_index index into fp8_unit_scale_gemm_num_algos's results.
 */
inline void fp8_unit_scale_gemm(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView w,
    tvm::ffi::TensorView workspace,
    int64_t algo_index) {
  CHECK_HOST(a.dim() == 2 && w.dim() == 2 && out.dim() == 2) << "a, w and out must be 2D";
  CHECK_HOST(a.stride(1) == 1 && a.stride(0) == a.size(1)) << "a must be contiguous";
  CHECK_HOST(w.stride(1) == 1 && w.stride(0) == w.size(1)) << "w must be contiguous [N, K]";
  CHECK_HOST(out.stride(1) == 1 && out.stride(0) == out.size(1)) << "out must be contiguous";
  CHECK_HOST(a.size(1) == w.size(1)) << "a and w must share K";
  CHECK_HOST(out.size(0) == a.size(0) && out.size(1) == w.size(0)) << "out must be [M, N]";
  CHECK_HOST(host::is_type<fp8_e4m3_t>(a.dtype()) && host::is_type<fp8_e4m3_t>(w.dtype())) << "a and w must be e4m3";
  CHECK_HOST(host::is_type<bf16_t>(out.dtype())) << "out must be bf16";

  const size_t workspace_bytes = static_cast<size_t>(workspace.numel());
  const auto& p = fp8_cublaslt_impl::problem(a.device().device_id, a.size(0), w.size(0), a.size(1), workspace_bytes);
  CHECK_HOST(algo_index >= 0 && algo_index < p.num_algos) << "algo_index " << algo_index << " of " << p.num_algos;

  const float alpha = 1.0f, beta = 0.0f;
  const cudaStream_t stream = LaunchKernel::resolve_device(a.device());
  const cublasStatus_t st = cublasLtMatmul(
      fp8_cublaslt_impl::handle(),
      p.desc,
      &alpha,
      w.data_ptr(),
      p.layout_w,
      a.data_ptr(),
      p.layout_a,
      &beta,
      out.data_ptr(),
      p.layout_d,
      out.data_ptr(),
      p.layout_d,
      &p.algos[algo_index].algo,
      workspace.data_ptr(),
      workspace_bytes,
      stream);
  CHECK_HOST(st == CUBLAS_STATUS_SUCCESS) << "cublasLtMatmul failed with status " << static_cast<int>(st);
}

}  // namespace sglang
