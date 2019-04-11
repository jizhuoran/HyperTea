#include <algorithm>
#include <vector>

#include "hypertea/operators/base_conv_op.hpp"
#include "hypertea/util/im2col.hpp"
#include "hypertea/util/math_functions.hpp"

namespace hypertea {



// template <typename Dtype>
// void BaseConvolutionOp_CPU<Dtype>::forward_cpu_gemm(const Dtype* input,
//     const Dtype* weights, Dtype* output, bool skip_im2col) {
//   const Dtype* col_buff = input;
//   if (!is_1x1_) {
//     if (!skip_im2col) {
//       conv_im2col_cpu(input, col_buffer_);
//     }
//     col_buff = col_buffer_;
//   }
//   for (int g = 0; g < group_; ++g) {
//     hypertea_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
//         group_, conv_out_spatial_dim_, kernel_dim_,
//         (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
//         (Dtype)0., output + output_offset_ * g);
//   }
// }


// template <typename Dtype>
// void BaseConvolutionOp_CPU<Dtype>::backward_cpu_gemm(const Dtype* output,
//     const Dtype* weights, Dtype* input) {
//   Dtype* col_buff = col_buffer_;
//   if (is_1x1_) {
//     col_buff = input;
//   }
//   for (int g = 0; g < group_; ++g) {
//     hypertea_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
//         conv_out_spatial_dim_, conv_out_channels_ / group_,
//         (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
//         (Dtype)0., col_buff + col_offset_ * g);
//   }
//   if (!is_1x1_) {
//     conv_col2im_cpu(col_buff, input);
//   }
// }



// template <typename Dtype>
// void BaseConvolutionOp_CPU<Dtype>::forward_cpu_bias(Dtype* output,
//     const Dtype* bias) {
//   hypertea_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
//       out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_,
//       (Dtype)1., output);
// }

// INSTANTIATE_CLASS_CPU(BaseConvolutionOp_CPU);
// INSTANTIATE_CLASS_GPU(BaseConvolutionOp_GPU);



}  // namespace hypertea
