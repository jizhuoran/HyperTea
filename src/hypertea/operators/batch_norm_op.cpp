// #include <algorithm>
// #include <vector>
// #include <math.h>

#include "hypertea/operators/batch_norm_op.hpp"
#include "hypertea/common.hpp"

namespace hypertea {

template <typename Dtype>
void mean_var(
  TensorCPU<Dtype>& x, 
  TensorCPU<Dtype>& mean, TensorCPU<Dtype>& var, 
  int channels, int spatial_dim, float eps) {
  
  auto x_data = x.mutable_data();
  auto mean_data = mean.mutable_data();
  auto var_data = var.mutable_data();
  

  int nspatial_dim = x.count() / channels;

  Dtype p = 0;

  for (int c = 0; c < channels; ++c) {
    mean_data[c] = 0;
    var_data[c] = 0;
    for (int bs = 0; bs < (nspatial_dim / spatial_dim); ++bs) {
      for (int i = 0; i < spatial_dim; ++i) {
        p = x_data[bs * channels * spatial_dim + c * spatial_dim + i];
        mean_data[c] += p;
        var_data[c] += p*p;
      }
    }
    mean_data[c] /= nspatial_dim;
    var_data[c] /= nspatial_dim;
    var_data[c] = sqrt(var_data[c] - mean_data[c]*mean_data[c] + eps);
  }
}



template <typename Dtype>
void mean_var(
  const TensorGPU<Dtype>& x, 
  TensorGPU<Dtype>& mean,
  TensorGPU<Dtype>& var,
  int channels,
  int spatial_dim,
  float eps){
  
  int nspatial_dim = x.count() / channels;
  int cspatial_dim = spatial_dim * channels;

  auto data = x.mutable_data();
  auto mean_ = mean.mutable_data();
  auto var_ = var.mutable_data();

  Dtype alpha_ = to_dtype<Dtype>(1. / nspatial_dim);
  Dtype eps_ = to_dtype<Dtype>(eps);

  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    "average_channeled",
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&data),
      std::make_pair(sizeof(cl_mem), (void *)&mean_),
      std::make_pair(sizeof(cl_mem), (void *)&var_),
      std::make_pair(sizeof(cl_int), (void *)&spatial_dim),
      std::make_pair(sizeof(cl_int), (void *)&cspatial_dim),
      std::make_pair(sizeof(cl_int), (void *)&nspatial_dim),
      std::make_pair(sizeof(Dtype), (void *)&alpha_),
      std::make_pair(sizeof(Dtype), (void *)&eps_)

    },
    std::vector<size_t> {128, static_cast<size_t>(channels), 1},
    std::vector<size_t> {128, 1, 1}
  );

}


template void mean_var(
  const TensorGPU<float>& x, 
  TensorGPU<float>& mean,
  TensorGPU<float>& var,
  int channels,
  int spatial_dim,
  float eps
);


template void mean_var(
  const TensorGPU<half>& x, 
  TensorGPU<half>& mean,
  TensorGPU<half>& var,
  int channels,
  int spatial_dim,
  float eps
);


template<typename DeviceTensor>
DeviceTensor BatchNormOp<DeviceTensor>::operator()(DeviceTensor input) {

  DeviceTensor output = inplace_? input : input.duplicate();

  DeviceTensor variance(channels_);

  if (!use_global_stats_) {
    mean_var(input, *mean_, variance, channels_, spatial_dim_, eps_);
  } else {

    variance.copy_data(*variance_);
    variance += eps_;
    inplace_sqrt(variance);
    
  }


  inplace_channeled_sub(output, *mean_, channels_, spatial_dim_);
 

  if(weight_ != nullptr) {
    auto weight_with_var = *weight_ / variance;
    if (bias_ != nullptr) {
      inplace_channeled_scaladd(output, weight_with_var, *bias_, channels_, spatial_dim_);
    } else {
      inplace_channeled_scal(output, weight_with_var, channels_, spatial_dim_);
    }
  } else {
    inplace_inv(variance);
    inplace_channeled_scal(output, variance, channels_, spatial_dim_);
  }

  return output;

}


DEFINE_FORWARD_FUNC(BatchNormOp);

}  // namespace hypertea
