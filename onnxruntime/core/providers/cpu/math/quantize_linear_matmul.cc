// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/quantize_linear_matmul.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/helper.h"
#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"
#include "core/util/gemmlowp_common.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

#define ROUNDING_RIGHT_SHIFT(x, shift) \
    ({((x) >> (shift)) + \
        (((shift) == 0 ? 0 : (((x) >> ((shift)-1)) & 1)) & \
             ((((shift) <= 1 ? 0 : ((x) & ((1 << ((shift)-1)) - 1))) != 0) | (((x) >> (shift)) & 1)));})
    inline int8_t saturate(int32_t num, int shift, bool relu) {
        num = ROUNDING_RIGHT_SHIFT(num, shift);
        // Clip result
        num = num > SCHAR_MAX ? SCHAR_MAX : (num < SCHAR_MIN ? SCHAR_MIN : num);
        if (relu) {
            num = num < 0 ? 0 : num;
        }
        return num;
    }
    void mymatmul(int dimI, int dimJ, int dimK, const int8_t* in1, const int8_t* in2, int8_t* out,
                  int shift, bool relu, const int32_t* bias = nullptr) {
        for (int i = 0; i < dimI; i++) {
            for (int j = 0; j < dimJ; j++) {
                int32_t res = 0;
                for (int k = 0; k < dimK; k++) {
                    res += in1[i * dimK + k] * in2[k * dimJ + j];
                }
                out[i * dimJ + j] = saturate(res + (bias != nullptr ? bias[i * dimJ + j] : 0), shift, relu);
            }
        }
    }
    void NaiveCPUMultiplyi8i8_i8(int dimI, int dimJ, int dimK, const int8_t* in1, const int8_t* in2, int8_t* out,
                                 int divisor, bool relu, const int32_t* bias = nullptr) {
        bool isPowerOf2 = divisor && !(divisor & (divisor - 1));
        if (!isPowerOf2) {
            throw std::runtime_error("Divisor passed to systolic matmul must be power of 2");
        }
        int shift = sizeof(int)*8 - __builtin_clz(divisor) - 1;
        return mymatmul(dimI, dimJ, dimK, in1, in2, out, shift, relu, bias);
    }

// only register this operator if low precision computation is enabled.
ONNX_OPERATOR_KERNEL_EX(
    QLinearMatMul,
    kOnnxDomain,
    10,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>()),
    QLinearMatMul<uint8_t, uint8_t, uint8_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QLinearMatMul,
    kOnnxDomain,
    10,
    int8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int8_t>()),
    QLinearMatMul<int8_t, int8_t, int8_t>);


template <>
Status QLinearMatMul<uint8_t, uint8_t, uint8_t>::Compute(OpKernelContext* ctx) const {
  auto a = ctx->Input<Tensor>(0);
  auto b = ctx->Input<Tensor>(3);
  ORT_ENFORCE(a != nullptr && b != nullptr);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  // validate offsets
  auto a_offset = ctx->Input<Tensor>(2);
  auto b_offset = ctx->Input<Tensor>(5);
  auto y_offset = ctx->Input<Tensor>(7);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_offset),
              "QLinearMatmul : input zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(b_offset),
              "QLinearMatmul : weight zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(y_offset),
              "QLinearMatmul : result zero point must be a scalar or 1D tensor of size 1");

  // validate scale
  auto a_scale = ctx->Input<Tensor>(1);
  auto b_scale = ctx->Input<Tensor>(4);
  auto y_scale = ctx->Input<Tensor>(6);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_scale),
              "QLinearMatmul : input scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(b_scale),
              "QLinearMatmul : weight scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(y_scale),
              "QLinearMatmul : result scale must be a scalar or 1D tensor of size 1");

  auto a_scale_data = *(a_scale->template Data<float>());
  auto b_scale_data = *(b_scale->template Data<float>());
  auto y_scale_data = *(y_scale->template Data<float>());

  const float real_multiplier = (a_scale_data * b_scale_data) / y_scale_data;

#ifdef MLAS_SUPPORTS_GEMM_U8X8
  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));
  auto gemm_output_data = alloc->Alloc(SafeInt<size_t>(sizeof(int32_t)) *
                                       static_cast<size_t>(helper.M()) * static_cast<size_t>(helper.N()));
  BufferUniquePtr gemm_output_buffer(gemm_output_data, BufferDeleter(alloc));
  auto* gemm_output = static_cast<int32_t*>(gemm_output_buffer.get());
#else
  // Compute the fixed point multiplier and shift for requantizing with GEMMLOWP.
  int32_t integer_multiplier;
  int right_shift;
  QuantizeMultiplier(real_multiplier, &integer_multiplier, &right_shift);
#endif

  for (size_t i = 0; i < helper.OutputOffsets().size(); i++) {
#ifdef MLAS_SUPPORTS_GEMM_U8X8
    QGemmu8u8_s32(static_cast<int>(helper.M()),
                  static_cast<int>(helper.N()),
                  static_cast<int>(helper.K()),
                  a->template Data<uint8_t>() + helper.LeftOffsets()[i],
                  static_cast<int>(helper.K()),
                  *a_offset->template Data<uint8_t>(),
                  b->template Data<uint8_t>() + helper.RightOffsets()[i],
                  static_cast<int>(helper.N()),
                  *b_offset->template Data<uint8_t>(),
                  gemm_output,
                  static_cast<int>(helper.N()),
                  ctx->GetOperatorThreadPool());

    MlasRequantizeOutput(gemm_output,
                         y->template MutableData<uint8_t>() + helper.OutputOffsets()[i],
                         nullptr,
                         static_cast<size_t>(helper.M()),
                         static_cast<size_t>(helper.N()),
                         real_multiplier,
                         *y_offset->template Data<uint8_t>());
#else
    GemmlowpMultiplyu8u8_u8(a->template Data<uint8_t>() + helper.LeftOffsets()[i],
                            b->template Data<uint8_t>() + helper.RightOffsets()[i],
                            y->template MutableData<uint8_t>() + helper.OutputOffsets()[i],
                            *a_offset->template Data<uint8_t>(),
                            *b_offset->template Data<uint8_t>(),
                            *y_offset->template Data<uint8_t>(),
                            static_cast<int>(helper.M()),
                            static_cast<int>(helper.N()),
                            static_cast<int>(helper.K()),
                            integer_multiplier,
                            right_shift);
#endif
  }

  return Status::OK();
}

template <>
Status QLinearMatMul<int8_t, int8_t, int8_t>::Compute(OpKernelContext* ctx) const {
  auto a = ctx->Input<Tensor>(0);
  auto b = ctx->Input<Tensor>(3);
  ORT_ENFORCE(a != nullptr && b != nullptr);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  // validate offsets
  auto a_offset = ctx->Input<Tensor>(2);
  auto b_offset = ctx->Input<Tensor>(5);
  auto y_offset = ctx->Input<Tensor>(7);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_offset),
              "QLinearMatmul : input zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(b_offset),
              "QLinearMatmul : weight zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(y_offset),
              "QLinearMatmul : result zero point must be a scalar or 1D tensor of size 1");

  auto a_offset_data = *(a_offset->template Data<int8_t>());
  auto b_offset_data = *(b_offset->template Data<int8_t>());
  auto y_offset_data = *(y_offset->template Data<int8_t>());
  ORT_ENFORCE(a_offset_data == 0, "Systolic can only handle zero offset for a");
  ORT_ENFORCE(b_offset_data == 0, "Systolic can only handle zero offset for b");
  ORT_ENFORCE(y_offset_data == 0, "Systolic can only handle zero offset for y");

  // validate scale
  auto a_scale = ctx->Input<Tensor>(1);
  auto b_scale = ctx->Input<Tensor>(4);
  auto y_scale = ctx->Input<Tensor>(6);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_scale),
              "QLinearMatmul : input scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(b_scale),
              "QLinearMatmul : weight scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(y_scale),
              "QLinearMatmul : result scale must be a scalar or 1D tensor of size 1");

  auto a_scale_data = *(a_scale->template Data<float>());
  auto b_scale_data = *(b_scale->template Data<float>());
  auto y_scale_data = *(y_scale->template Data<float>());

  ORT_ENFORCE(a_scale_data != 0, "a_scale_data cannot be 0");
  ORT_ENFORCE(b_scale_data != 0, "b_scale_data cannot be 0");
  ORT_ENFORCE(y_scale_data != 0, "y_scale_data cannot be 0");


  //const float real_multiplier = (a_scale_data * b_scale_data) / y_scale_data;
  unsigned int rounded_divisor = nearestPowerOfTwo(y_scale_data / (a_scale_data * b_scale_data));
  for (size_t i = 0; i < helper.OutputOffsets().size(); i++) {
      NaiveCPUMultiplyi8i8_i8(
                            static_cast<int>(helper.M()),
                            static_cast<int>(helper.N()),
                            static_cast<int>(helper.K()),
                            a->template Data<int8_t>() + helper.LeftOffsets()[i],
                            b->template Data<int8_t>() + helper.RightOffsets()[i],
                            y->template MutableData<int8_t>() + helper.OutputOffsets()[i],
                            rounded_divisor, //real_multiplier);
                            /*relu= */ false);
  }

  return Status::OK();
}
}  // namespace onnxruntime
