#include <algorithm>
#include <vector>
#include <math.h>

#include "hypertea/operators/batch_norm_op.hpp"
#include "hypertea/util/math_functions.hpp"

namespace hypertea {

template <>
TensorCPU<float> BatchNormOp_CPU<float>::Forward(TensorCPU<float> &input_tensor) {

  const float* input_data = input_tensor.immutable_data();
  float* output_data = inplace_? input_tensor.mutable_data() : new float[input_tensor.count()];

  if (!inplace_) {
    hypertea_copy(top_count_, input_data, output_data);
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
        1. / (num_ * spatial_dim_), input_data,
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
      spatial_sum_multiplier_, 1., output_data);

  if (!use_global_stats_) {
    // compute variance using var(X) = E((X-EX)^2)
    hypertea_sqr<float>(top_count_, output_data,
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
  hypertea_div(top_count_, output_data, temp_, output_data);



  if(weight_ != NULL) {

    input_data = inplace_? input_data : output_data;

    float* output_data_ptr_keeper = output_data;

    int inner_dim = top_count_ / channels_;

    for (int n = 0; n < num_; ++n) {
      for (int d = 0; d < channels_; ++d) {
        const float factor = weight_[d];
        hypertea_cpu_scale(inner_dim, factor, input_data, output_data);
        input_data += inner_dim;
        output_data += inner_dim;
      } 
    }

    if (bias_ != NULL) {

      output_data = output_data_ptr_keeper;

      for (int n = 0; n < num_; ++n) {

        hypertea_cpu_gemm(CblasNoTrans, CblasNoTrans, channels_,
            inner_dim, 1, float(1), bias_,
            bias_multiplier_, float(1), output_data);
        output_data += (channels_ * inner_dim); 
      }
    }

    output_data = output_data_ptr_keeper;
  }


  return inplace_? input_tensor:TensorCPU<float>(output_data, input_tensor.size());  

}

#ifdef USE_OPENCL

template <typename Dtype>
TensorGPU<Dtype> BatchNormOp_GPU<Dtype>::Forward(TensorGPU<Dtype> input_tensor){

  const cl_mem input_data = input_tensor.immutable_data();
  TensorGPU<Dtype> output_tensor = inplace_? input_tensor : TensorGPU<Dtype>(input_tensor.count());
  cl_mem output_data = output_tensor.mutable_data();


  float sum_shift_num = 1.;//64.0;
  float top_shift_num = 1.;//32.0;

  if (input_data != output_data) {
    hypertea_cl_copy<Dtype>(top_count_, input_data, output_data);
  }

  if (use_global_stats_) {
    // use the stored mean/variance estimates.
    hypertea_gpu_scale<Dtype>(channels_, scale_factor_,
        mean_, mean_);
    hypertea_gpu_scale<Dtype>(channels_, scale_factor_,
        variance_, variance_);
  } else {
    // compute mean

    hypertea_gpu_bsum<Dtype>(channels_ * num_, spatial_dim_, input_data, 
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
      spatial_sum_multiplier_, float(1.), output_data);


  if (!use_global_stats_) {


    hypertea_gpu_scal<Dtype>(top_count_, 1/top_shift_num, output_data);

    // compute variance using var(X) = E((X-EX)^2)
    hypertea_gpu_mul<Dtype>(top_count_, output_data, output_data,
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

  hypertea_gpu_scal<Dtype>(top_count_, top_shift_num, output_data);
  hypertea_gpu_scal<Dtype>(channels_, top_shift_num, variance_);

  // replicate variance to input size
  hypertea_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, float(1),
      batch_sum_multiplier_, variance_, float(0.),
      num_by_chans_);
  hypertea_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num_,
      spatial_dim_, 1, float(1.), num_by_chans_,
      spatial_sum_multiplier_, float(0.), temp_);

  hypertea_gpu_div<Dtype>(top_count_, output_data, temp_, output_data);

  if (weight_ != NULL) {

    int inner_dim = top_count_ / channels_;


    if (bias_ != NULL) {

      cl_int ret;

      cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "ScaleBiasForward", &ret);
      OPENCL_CHECK(ret);


      // Set arguments for kernel
      OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&output_data));  
      OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output_data));  
      OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&top_count_));  
      OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&weight_)); 
      OPENCL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&bias_));   
      OPENCL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&channels_));  
      OPENCL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&inner_dim));  

      size_t global_size = HYPERTEA_GET_BLOCKS(top_count_);
      
      OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  


    } else {


      cl_int ret;

      cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "ScaleForward", &ret);
      OPENCL_CHECK(ret);

      // Set arguments for kernel
      OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&output_data));  
      OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output_data));  
      OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&top_count_));  
      OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&weight_)); 
      OPENCL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&channels_));  
      OPENCL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&inner_dim));  

      size_t global_size = HYPERTEA_GET_BLOCKS(top_count_);
      
      OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  

    }
  }
  
  return output_tensor;

}

#endif //USE_OPENCL

/*

template <typename scalar_t, typename accscalar_t, bool train, typename index_t>
__global__ void batch_norm_transform_input_kernel(
    const PackedTensorAccessor<scalar_t, 3, RestrictPtrTraits, index_t> input,
    PackedTensorAccessor<scalar_t, 3, RestrictPtrTraits, index_t> output,
    const PackedTensorAccessor<typename std::conditional<train, accscalar_t, scalar_t>::type, 1, RestrictPtrTraits, index_t> mean_,
    const PackedTensorAccessor<typename std::conditional<train, accscalar_t, scalar_t>::type, 1, RestrictPtrTraits, index_t> var_or_invstd,
    const PackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, index_t> weight,
    const PackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, index_t> bias,
    accscalar_t epsilon) {

  index_t plane = blockIdx.x;

  if (plane >= input.size(1)) {
    return;
  }

  accscalar_t gamma = weight.size(0) > 0 ? static_cast<accscalar_t>(weight[plane]) : static_cast<accscalar_t>(1);
  accscalar_t beta = bias.size(0) > 0 ? static_cast<accscalar_t>(bias[plane]) : static_cast<accscalar_t>(0);
  accscalar_t mean = static_cast<accscalar_t>(mean_[plane]);
  accscalar_t invstd;
  if (train) {
    invstd = var_or_invstd[plane];
  } else {
    invstd = static_cast<accscalar_t>(1) / device_sqrt(static_cast<accscalar_t>(var_or_invstd[plane]) + epsilon);
  }

  index_t bs = input.size(0);
  index_t fs = input.size(2);

  index_t bstep  = blockDim.y * gridDim.y;
  for (index_t batch = threadIdx.y + blockIdx.y * blockDim.y; batch < bs; batch += bstep) {
    auto o = output[batch][plane];
    auto i = input[batch][plane];
    for (index_t feature = threadIdx.x; feature < fs; feature += blockDim.x) {
      o[feature] = static_cast<scalar_t>(gamma * (i[feature] - mean) * invstd + beta);
    }
  }
}

*/


INSTANTIATE_CLASS_CPU(BatchNormOp_CPU);
INSTANTIATE_CLASS_GPU(BatchNormOp_GPU);
// REGISTER_LAYER_CLASS(BatchNorm);
}  // namespace hypertea
