#ifndef HYPERTEA_UTIL_IM2COL_HPP_
#define HYPERTEA_UTIL_IM2COL_HPP_

#include "hypertea/tensor.hpp"

namespace hypertea {

template <typename Dtype>
void im2col(const TensorCPU<Dtype>& data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    TensorCPU<Dtype>& data_col);

template <typename Dtype>
void col2im(const TensorCPU<Dtype>& data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    TensorCPU<Dtype>& data_im);

#ifdef USE_OPENCL
template <typename Dtype>
void im2col(const TensorGPU<Dtype>& data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    TensorGPU<Dtype>& data_col);


template <typename Dtype>
void col2im(const TensorGPU<Dtype>& data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    TensorGPU<Dtype>& data_im);
#endif //USE_OPENCL


}  // namespace hypertea

#endif  // HYPERTEA_UTIL_IM2COL_HPP_
