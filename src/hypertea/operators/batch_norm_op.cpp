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
    // hypertea_cpu_scale(channels_, scale_factor_,
    //     mean_, mean_);
    // hypertea_cpu_scale(channels_, scale_factor_,
    //     variance_, variance_);
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

  // std::cout << "pass this line!!" << std::endl;

  // subtract mean
  hypertea_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, 1,
      batch_sum_multiplier_, mean_, 0.,
      num_by_chans_);
  hypertea_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, channels_ * num_,
      spatial_dim_, 1, -1, num_by_chans_,
      spatial_sum_multiplier_, 1., output_data);

  // std::cout << "pass this line1!!" << std::endl;



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

  // std::cout << "pass this line2!!" << std::endl;


  if(weight_ != NULL) {

    input_data = inplace_? input_data : output_data;

    float* output_data_ptr_keeper = output_data;

    int inner_dim = top_count_ / (channels_ * num_);

    for (int n = 0; n < num_; ++n) {
      for (int d = 0; d < channels_; ++d) {
        const float factor = weight_[d];
        hypertea_cpu_scale(inner_dim, factor, input_data, output_data);
        input_data += inner_dim;
        output_data += inner_dim;
      } 
    }

    // std::cout << "pass this line3!!" << std::endl;

    if (bias_ != NULL) {

      output_data = output_data_ptr_keeper;

      // std::cout << "pass this line4!!" << std::endl;

      for (int n = 0; n < num_; ++n) {

        // std::cout << "pass this line4." << n << "!!" <<std::endl;


        hypertea_cpu_gemm(CblasNoTrans, CblasNoTrans, channels_,
            inner_dim, 1, float(1), bias_,
            bias_multiplier_, float(1), output_data);
        output_data += (channels_ * inner_dim); 
      }
    }

    // std::cout << "pass this line5!!" << std::endl;


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


  if (!inplace_) {
    output_tensor.copy_data(input_tensor);
  }

  if (!use_global_stats_) {
    // compute mean

    hypertea_gpu_bsum<Dtype>(channels_ * num_, spatial_dim_, input_data, 
                          1/sum_shift_num_, (sum_shift_num_*sum_shift_num_)/(num_ * spatial_dim_), 
                          num_by_chans_, 1);
    
    hypertea_gpu_gemv<Dtype>(CblasTrans, num_, channels_, float(1.),
        num_by_chans_, batch_sum_multiplier_, float(0.), tmean_.mutable_data());

  }


  inplace_channeled_sub<Dtype>(output_tensor, tmean_, channels_, spatial_dim_);


  if (!use_global_stats_) {

    // compute variance using var(X) = E((X-EX)^2)

    output_tensor *= static_cast<float>(1/top_shift_num_);
    ttemp_ = output_tensor * output_tensor;

    hypertea_gpu_bsum<Dtype>(channels_ * num_, spatial_dim_, ttemp_.mutable_data(), 
                          1/sum_shift_num_, (sum_shift_num_*sum_shift_num_) / (num_ * spatial_dim_), 
                          num_by_chans_, 1);

    hypertea_gpu_gemv<Dtype>(CblasTrans, num_, channels_, float(1.0),
        num_by_chans_, batch_sum_multiplier_, float(0.),
        tvariance_.mutable_data());  // E((X_EX)^2)

  }

  tvariance_ += eps_;
  tvariance_.sqrt();
  // normalize variance

  output_tensor *= top_shift_num_;
  tvariance_ *= top_shift_num_;


  if(affine) {
    auto weight_with_var = tweight_ / tvariance_;
    if (bias_ != NULL) {
      inplace_channeled_scaladd(output_tensor, weight_with_var, tbias_, channels_, spatial_dim_);
    } else {
      inplace_channeled_scal(output_tensor, weight_with_var, channels_, spatial_dim_);
    }
  } else {
    hypertea_gpu_inv<Dtype>(channels_, tvariance_.mutable_data(), tvariance_.mutable_data());
    inplace_channeled_scal(output_tensor, tvariance_, channels_, spatial_dim_);
  }


  
  return output_tensor;

}

#endif //USE_OPENCL



INSTANTIATE_CLASS_CPU(BatchNormOp_CPU);
INSTANTIATE_CLASS_GPU(BatchNormOp_GPU);
// REGISTER_LAYER_CLASS(BatchNorm);
}  // namespace hypertea
