#include <algorithm>
#include <vector>
#include <math.h>

#include "hypertea/operators/batch_norm_op.hpp"
#include "hypertea/util/math_functions.hpp"

namespace hypertea {


template <>
void BatchNormOp_CPU<float>::Forward(const std::vector<float*> bottom_datas,
      const std::vector<float*> top_datas) {

  float* bottom_data = bottom_datas[0];
  float* top_data = top_datas[0];


  if (bottom_data != top_data) {
    hypertea_copy(top_count_, bottom_data, top_data);
  }

  if (use_global_stats_) {
    // use the stored mean/variance estimates.
    hypertea_cpu_scale(channels_, scale_factor_,
        mean_, mean_);
    hypertea_cpu_scale(channels_, scale_factor_,
        variance_, variance_);
  } else {
    // compute mean
    hypertea_cpu_gemv<float>(CblasNoTrans, channels_ * num_, spatial_dim_,
        1. / (num_ * spatial_dim_), bottom_data,
        spatial_sum_multiplier_, 0.,
        num_by_chans_);
    hypertea_cpu_gemv<float>(CblasTrans, num_, channels_, 1.,
        num_by_chans_, batch_sum_multiplier_, 0.,
        mean_);
  }

  // subtract mean
  hypertea_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, 1,
      batch_sum_multiplier_, mean_, 0.,
      num_by_chans_);
  hypertea_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, channels_ * num_,
      spatial_dim_, 1, -1, num_by_chans_,
      spatial_sum_multiplier_, 1., top_data);

  if (!use_global_stats_) {
    // compute variance using var(X) = E((X-EX)^2)
    hypertea_sqr<float>(top_count_, top_data,
                     temp_);  // (X-EX)^2
    hypertea_cpu_gemv<float>(CblasNoTrans, channels_ * num_, spatial_dim_,
        1. / (num_ * spatial_dim_), temp_,
        spatial_sum_multiplier_, 0.,
        num_by_chans_);
    hypertea_cpu_gemv<float>(CblasTrans, num_, channels_, 1.,
        num_by_chans_, batch_sum_multiplier_, 0.,
        variance_);  // E((X_EX)^2)

  }

  // normalize variance
  hypertea_add_scalar(channels_, eps_, variance_);
  hypertea_sqrt(channels_, variance_, variance_);

  // replicate variance to input size
  hypertea_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, 1,
      batch_sum_multiplier_, variance_, 0.,
      num_by_chans_);
  hypertea_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, channels_ * num_,
      spatial_dim_, 1, 1., num_by_chans_,
      spatial_sum_multiplier_, 0., temp_);
  hypertea_div(top_count_, top_data, temp_, top_data);

}


#ifdef USE_OPENCL

template <typename Dtype>
void BatchNormOp_GPU<Dtype>::Forward(const std::vector<cl_mem> bottom_datas,
      const std::vector<cl_mem> top_datas) {


  float sum_shift_num = 1.;//64.0;
  float top_shift_num = 1.;//32.0;

  if (bottom_datas[0] != top_datas[0]) {
    hypertea_cl_copy<Dtype>(top_count_, bottom_datas[0], top_datas[0]);
  }

  if (use_global_stats_) {
    // use the stored mean/variance estimates.
    hypertea_gpu_scale<Dtype>(channels_, scale_factor_,
        mean_, mean_);
    hypertea_gpu_scale<Dtype>(channels_, scale_factor_,
        variance_, variance_);
  } else {
    // compute mean

    hypertea_gpu_bsum<Dtype>(channels_ * num_, spatial_dim_, bottom_datas[0], 
                          1/sum_shift_num, (sum_shift_num*sum_shift_num)/(num_ * spatial_dim_), 
                          num_by_chans_, 1);

    hypertea_gpu_gemv<Dtype>(CblasTrans, num_, channels_, float(1.),
        num_by_chans_, batch_sum_multiplier_, float(0.),
        mean_);
  }

  // subtract mean
  hypertea_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, float(1),
      batch_sum_multiplier_, mean_, float(0.),
      num_by_chans_);
  

  hypertea_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num_,
      spatial_dim_, 1, float(-1), num_by_chans_,
      spatial_sum_multiplier_, float(1.), top_datas[0]);


  if (!use_global_stats_) {


    hypertea_gpu_scal<Dtype>(top_count_, 1/top_shift_num, top_datas[0]);

    // compute variance using var(X) = E((X-EX)^2)
    hypertea_gpu_mul<Dtype>(top_count_, top_datas[0], top_datas[0],
        temp_);  // (X-EX)^2

    hypertea_gpu_bsum<Dtype>(channels_ * num_, spatial_dim_, temp_, 
                          1/sum_shift_num, (sum_shift_num*sum_shift_num) / (num_ * spatial_dim_), 
                          num_by_chans_, 1);

    hypertea_gpu_gemv<Dtype>(CblasTrans, num_, channels_, float(1.0),
        num_by_chans_, batch_sum_multiplier_, float(0.),
        variance_);  // E((X_EX)^2)

  }


  // normalize variance
  hypertea_gpu_add_scalar<Dtype>(channels_, eps_, variance_);
  hypertea_gpu_sqrt<Dtype>(channels_, variance_, variance_);

  hypertea_gpu_scal<Dtype>(top_count_, top_shift_num, top_datas[0]);
  hypertea_gpu_scal<Dtype>(channels_, top_shift_num, variance_);

  // replicate variance to input size
  hypertea_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, float(1),
      batch_sum_multiplier_, variance_, float(0.),
      num_by_chans_);
  hypertea_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num_,
      spatial_dim_, 1, float(1.), num_by_chans_,
      spatial_sum_multiplier_, float(0.), temp_);

  hypertea_gpu_div<Dtype>(top_count_, top_datas[0], temp_, top_datas[0]);

}

#endif //USE_OPENCL



INSTANTIATE_CLASS_CPU(BatchNormOp_CPU);
INSTANTIATE_CLASS_GPU(BatchNormOp_GPU);
// REGISTER_LAYER_CLASS(BatchNorm);
}  // namespace hypertea
