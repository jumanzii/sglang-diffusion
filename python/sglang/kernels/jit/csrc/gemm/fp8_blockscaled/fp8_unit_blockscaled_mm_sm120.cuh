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

/* FP8 e4m3 GEMM for SM120 on the block-scaled (MXFP8) tensor-core path, with every
UE8M0 scale factor set to 2^0, and a pluggable CUTLASS epilogue.

Why block-scaled with unit scales: on GeForce SM120 parts (RTX 5090) the plain FP8
mma.sync with fp32 accumulation (SASS QMMA.16832.F32.E4M3.E4M3, which CUTLASS's
OpClassTensorOp FP8 path emits) runs at half the rate of the block-scaled form
(QMMA.SF...E8, mma.sync.kind::mxf8f6f4.block_scale). cuBLASLt's own FP8 kernels on
this part issue the block-scaled form with unit scales. With every scale factor
equal to 1 the product is exactly the plain FP8 product, so any epilogue written
for a plain FP8 GEMM applies unchanged.

Epilogue hook: the kernel is templated on an epilogue policy, a struct with
  - `Params`: the host-side arguments of the epilogue (plain pointers and scalars);
  - `template <class TileShape, class Schedule> using Collective`: the CUTLASS
    collective epilogue, for the pingpong (TmaWarpSpecialized) or cooperative
    (TmaWarpSpecializedCooperative) schedule;
  - `template <class Collective, class TileShape> static Collective::Arguments
    arguments(Params const&, ElementD* d, Collective::StrideD)`.
An element-wise epilogue is an Sm90 fusion (EVT) tree: write a struct with
`Params`, `Fusion<TileShape>` and `fusion_args<TileShape>(Params)` and use
EvtEpilogue<that struct> as the policy, as PerTokenPerChannelScale and Unscaled do. An epilogue that is not
element-wise, such as a per-row normalization over the tile's columns, supplies its own Collective: with the pingpong
schedule one consumer warp group holds the whole 128x128 fp32 tile in registers. A new fused GEMM adds a policy and an
exported entry point; the mainloop is shared.
*/

#pragma once

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstdint>
#include <cuda_runtime.h>

// clang-format off
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"
// clang-format on

namespace sglang {

using namespace host;
using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

namespace fp8_unit_blockscaled {

using ElementD = cutlass::bfloat16_t;
constexpr int kAlignD = 128 / cutlass::sizeof_bits<ElementD>::value;

/// \brief The collective epilogue for an Sm90 fusion (EVT) tree that stores bf16 D, row major.
template <class TileShape, class Schedule, class Fusion>
using EvtCollective = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    cutlass::arch::OpClassTensorOp,
    TileShape,
    Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float,
    float,
    void,
    cutlass::layout::RowMajor,
    kAlignD,
    ElementD,
    cutlass::layout::RowMajor,
    kAlignD,
    Schedule,
    Fusion>::CollectiveOp;

/// \brief Epilogue policy for an element-wise epilogue. `Evt` defines `Params`,
/// `template <class TileShape> using Fusion` (an Sm90 EVT tree) and
/// `template <class TileShape> static Fusion<TileShape>::Arguments fusion_args(Params const&)`.
template <class Evt>
struct EvtEpilogue {
  using Params = typename Evt::Params;

  template <class TileShape, class Schedule>
  using Collective = EvtCollective<TileShape, Schedule, typename Evt::template Fusion<TileShape>>;

  template <class Collective, class TileShape>
  static typename Collective::Arguments
  arguments(Params const& params, ElementD* d, typename Collective::StrideD stride_d) {
    return {Evt::template fusion_args<TileShape>(params), nullptr, stride_d, d, stride_d};
  }
};

/// \brief D = bf16(scale_a[m] * (scale_b[n] * acc)), one rounding.
/// Same math as sgl-kernel's DeviceGemmFp8RowwiseSm120.
struct PerTokenPerChannelScaleEvt {
  struct Params {
    const float* scale_a;  // M per-token scales
    const float* scale_b;  // N per-channel scales
  };

  template <class TileShape>
  struct Types {
    using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
    using ScaleA =
        cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, float, float, Stride<Int<1>, Int<0>, Int<0>>>;
    using ScaleB =
        cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, float, float, Stride<Int<0>, Int<1>, Int<0>>>;
    using Mul = cutlass::epilogue::fusion::
        Sm90Compute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using MulOut = cutlass::epilogue::fusion::
        Sm90Compute<cutlass::multiplies, ElementD, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using Inner = cutlass::epilogue::fusion::Sm90EVT<Mul, ScaleB, Accum>;
    using Fusion = cutlass::epilogue::fusion::Sm90EVT<MulOut, ScaleA, Inner>;
  };

  template <class TileShape>
  using Fusion = typename Types<TileShape>::Fusion;

  template <class TileShape>
  static typename Fusion<TileShape>::Arguments fusion_args(Params const& p) {
    typename Types<TileShape>::Inner::Arguments inner{{p.scale_b}, {}, {}};
    return {{p.scale_a}, inner, {}};
  }
};

/// \brief D = bf16(acc). The caller applies its scales later
/// (the deferred-scale consumers of apply_fp8_linear_deferred_scale).
struct UnscaledEvt {
  struct Params {};

  template <class TileShape>
  using Fusion = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::
          Sm90Compute<cutlass::epilogue::thread::Identity, ElementD, float, cutlass::FloatRoundStyle::round_to_nearest>,
      cutlass::epilogue::fusion::Sm90AccFetch>;

  template <class TileShape>
  static typename Fusion<TileShape>::Arguments fusion_args(Params const&) {
    return {{}, {}};
  }
};

using PerTokenPerChannelScale = EvtEpilogue<PerTokenPerChannelScaleEvt>;
using Unscaled = EvtEpilogue<UnscaledEvt>;

/// \brief The CUTLASS kernel for one tile shape, schedule and epilogue policy.
template <class Epilogue, int kTileM, int kTileN, int kTileK, bool kPingpong, bool kStreamK>
struct Gemm {
  using ElementAB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using ElementData = typename ElementAB::DataType;
  using ElementSF = typename ElementAB::ScaleFactorType;
  using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
  using ClusterShape = Shape<_1, _1, _1>;

  static constexpr int kAlignAB = 128 / cutlass::sizeof_bits<ElementData>::value;

  using MainloopSchedule = std::conditional_t<
      kPingpong,
      cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
      cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120>;
  using EpilogueSchedule = std::
      conditional_t<kPingpong, cutlass::epilogue::TmaWarpSpecialized, cutlass::epilogue::TmaWarpSpecializedCooperative>;

  using CollectiveEpilogue = typename Epilogue::template Collective<TileShape, EpilogueSchedule>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120,
      cutlass::arch::OpClassBlockScaledTensorOp,
      ElementAB,
      cutlass::layout::RowMajor,
      kAlignAB,
      ElementAB,
      cutlass::layout::ColumnMajor,
      kAlignAB,
      float,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;

  // StreamK keeps CUTLASS's default deterministic reduction, so results are reproducible run to run.
  using TileScheduler = std::conditional_t<kStreamK, cutlass::gemm::StreamKScheduler, void>;
  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, TileScheduler>;
  using Device = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using SfConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
};

/// \brief Bytes of UE8M0 scale factors one operand of `rows` x `k` needs: the SF
/// layout pads rows to 128 and K to 4 scale factors of 32 elements each.
SGL_DEVICE_HOST constexpr int64_t unit_sf_bytes(int64_t rows, int64_t k) {
  return (rows + 127) / 128 * 128 * ((k + 127) / 128 * 4);
}

template <class Epilogue, class Cfg>
void run(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView unit_sf,
    typename Epilogue::Params const& epilogue_params,
    int64_t max_swizzle) {
  using GemmKernel = typename Cfg::GemmKernel;

  auto M = SymbolicSize{"m"};
  auto N = SymbolicSize{"n"};
  auto K = SymbolicSize{"k"};
  auto lda = SymbolicSize{"lda"};
  auto ldd = SymbolicSize{"ldd"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  // TMA needs 16-byte aligned base pointers and row pitches.
  TensorMatcher({M, K})
      .with_strides({lda, 1})
      .with_dtype<fp8_e4m3_t>()
      .with_device<kDLCUDA>(device)
      .ensure_alignment(16)
      .verify(a);
  TensorMatcher({N, K}).with_dtype<fp8_e4m3_t>().with_device<kDLCUDA>(device).ensure_alignment(16).verify(b);
  TensorMatcher({M, N})
      .with_strides({ldd, 1})
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device)
      .ensure_alignment(16)
      .verify(out);
  const int64_t m = M.unwrap();
  const int64_t n = N.unwrap();
  const int64_t k = K.unwrap();
  CHECK_HOST(k % 16 == 0) << "K must be a multiple of 16, got " << k;
  CHECK_HOST(n % 8 == 0) << "N must be a multiple of 8, got " << n;
  auto sf_bytes = SymbolicSize{"sf_bytes"};
  TensorMatcher({sf_bytes}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(unit_sf);
  const int64_t sf_needed = unit_sf_bytes(m > n ? m : n, k);
  CHECK_HOST(sf_bytes.unwrap() >= sf_needed) << "unit_sf holds " << sf_bytes.unwrap() << " bytes, need " << sf_needed;

  const auto m32 = static_cast<int32_t>(m);
  const auto n32 = static_cast<int32_t>(n);
  const auto k32 = static_cast<int32_t>(k);
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideD = typename GemmKernel::StrideD;
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m32, k32, 1));
  cute::get<0>(stride_a) = lda.unwrap();
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n32, k32, 1));
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m32, n32, 1));
  cute::get<0>(stride_d) = ldd.unwrap();
  auto layout_sfa = Cfg::SfConfig::tile_atom_to_shape_SFA(make_shape(m32, n32, k32, 1));
  auto layout_sfb = Cfg::SfConfig::tile_atom_to_shape_SFB(make_shape(m32, n32, k32, 1));
  // Both operands read the same all-unit buffer; it only has to cover the larger one.
  const auto* sf = static_cast<typename Cfg::ElementSF const*>(unit_sf.data_ptr());

  auto* ptr_d = static_cast<ElementD*>(out.data_ptr());
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m32, n32, k32, 1},
      {static_cast<typename Cfg::ElementData const*>(a.data_ptr()),
       stride_a,
       static_cast<typename Cfg::ElementData const*>(b.data_ptr()),
       stride_b,
       sf,
       layout_sfa,
       sf,
       layout_sfb},
      Epilogue::template arguments<typename Cfg::CollectiveEpilogue, typename Cfg::TileShape>(
          epilogue_params, ptr_d, stride_d),
  };
  const DLDevice dev = device.unwrap();
  args.hw_info.device_id = dev.device_id;
  args.hw_info.sm_count = static_cast<int>(host::runtime::get_sm_count(dev.device_id));
  args.scheduler.max_swizzle_size = static_cast<int>(max_swizzle);

  typename Cfg::Device gemm_op;
  cutlass::Status status = gemm_op.can_implement(args);
  CHECK_HOST(status == cutlass::Status::kSuccess) << cutlassGetStatusString(status);
  const size_t workspace_size = Cfg::Device::get_workspace_size(args);
  auto workspace_tensor = host::ffi::alloc_workspace_tensor(workspace_size, dev);
  void* workspace = workspace_size == 0 ? nullptr : workspace_tensor.data_ptr();
  const cudaStream_t stream = LaunchKernel::resolve_device(dev);
  status = gemm_op.initialize(args, workspace, stream);
  CHECK_HOST(status == cutlass::Status::kSuccess) << cutlassGetStatusString(status);
  status = gemm_op.run(stream);
  CHECK_HOST(status == cutlass::Status::kSuccess) << cutlassGetStatusString(status);
}

}  // namespace fp8_unit_blockscaled

/**
 * \brief out[M, N] = bf16(scale_a[M] * scale_b[N] * (a[M, K] @ b[N, K]^T)) on SM120, e4m3 inputs.
 * \tparam kTileM, kTileN, kTileK CTA tile shape.
 * \tparam kPingpong two consumer warp groups on alternate tiles instead of one cooperative tile.
 * \tparam kStreamK StreamK tile scheduler instead of the persistent data-parallel one.
 * \param out [M, N] bf16; rows may be strided (16-byte aligned pitch).
 * \param a [M, K] e4m3; rows may be strided (16-byte aligned pitch).
 * \param b [N, K] e4m3, contiguous (the stored weight).
 * \param unit_sf uint8 buffer of 0x7f (UE8M0 2^0), at least unit_sf_bytes(max(M, N), K) long.
 * \param scales_a M fp32 per-token scales.
 * \param scales_b N fp32 per-channel scales.
 * \param max_swizzle the tile scheduler's max swizzle size (1, 2, 4 or 8).
 */
template <int kTileM, int kTileN, int kTileK, bool kPingpong, bool kStreamK>
void fp8_unit_blockscaled_scaled_mm_sm120(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView unit_sf,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    int64_t max_swizzle) {
  using Epilogue = fp8_unit_blockscaled::PerTokenPerChannelScale;
  using Cfg = fp8_unit_blockscaled::Gemm<Epilogue, kTileM, kTileN, kTileK, kPingpong, kStreamK>;
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({a.size(0)}).with_dtype<float>().with_device<kDLCUDA>(device).verify(scales_a);
  TensorMatcher({b.size(0)}).with_dtype<float>().with_device<kDLCUDA>(device).verify(scales_b);
  const Epilogue::Params params{
      static_cast<const float*>(scales_a.data_ptr()), static_cast<const float*>(scales_b.data_ptr())};
  fp8_unit_blockscaled::run<Epilogue, Cfg>(out, a, b, unit_sf, params, max_swizzle);
}

/**
 * \brief out[M, N] = bf16(a[M, K] @ b[N, K]^T) on SM120, e4m3 inputs, no scales.
 * Arguments as fp8_unit_blockscaled_scaled_mm_sm120, without the scales.
 */
template <int kTileM, int kTileN, int kTileK, bool kPingpong, bool kStreamK>
void fp8_unit_blockscaled_mm_sm120(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView unit_sf,
    int64_t max_swizzle) {
  using Epilogue = fp8_unit_blockscaled::Unscaled;
  using Cfg = fp8_unit_blockscaled::Gemm<Epilogue, kTileM, kTileN, kTileK, kPingpong, kStreamK>;
  fp8_unit_blockscaled::run<Epilogue, Cfg>(out, a, b, unit_sf, {}, max_swizzle);
}

#endif  // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

}  // namespace sglang
