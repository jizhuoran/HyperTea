#include <algorithm>
#include <vector>

#include "hypertea/operators/scale_op.hpp"

namespace hypertea {

template<typename DeviceTensor>
DeviceTensor ScaleOp<DeviceTensor>::operator()(DeviceTensor& input) {

  DeviceTensor output = inplace_? input : input.duplicate();

  if (has_bias_) {
      inplace_channeled_scaladd(output, *weight_, *bias_, channels_, spatial_dim_);
  } else {
      inplace_channeled_scal(output, *weight_, channels_, spatial_dim_);
  }

  return output;
}
DEFINE_FORWARD_FUNC(ScaleOp);



template <>
TensorCPU<float> ScaleOp_CPU<float>::Forward(TensorCPU<float> &input_tensor) {


  const float* input_data = input_tensor.immutable_data();
  float* output_data = inplace_? input_tensor.mutable_data() : new float[input_tensor.size()];

  float* output_data_ptr_keeper = output_data;


  for (int n = 0; n < outer_dim_; ++n) {
    for (int d = 0; d < scale_dim_; ++d) {
      const float factor = scale_data_[d];
      hypertea_cpu_scale(inner_dim_, factor, input_data, output_data);
      input_data += inner_dim_;
      output_data += inner_dim_;
    } 
  } 

  if (bias_data_ != NULL) {

    output_data = output_data_ptr_keeper;

    for (int n = 0; n < outer_dim_; ++n) {

      hypertea_cpu_gemm(CblasNoTrans, CblasNoTrans, scale_dim_,
          inner_dim_, 1, float(1), bias_data_,
          bias_multiplier_, float(1), output_data);
      output_data += (scale_dim_ * inner_dim_); 
    }
  }

  return inplace_? input_tensor:TensorCPU<float>(output_data_ptr_keeper, input_tensor.size());  

}

#ifdef USE_OPENCL


template <typename Dtype>
TensorGPU<Dtype> ScaleOp_GPU<Dtype>::Forward(TensorGPU<Dtype> input_tensor){

  TensorGPU<Dtype> output_tensor = inplace_? input_tensor : input_tensor.duplicate();

  if (has_bias_) {
      inplace_channeled_scaladd(output_tensor, weight_, bias_, scale_dim_, inner_dim_);
  } else {
      inplace_channeled_scal(output_tensor, weight_, scale_dim_, inner_dim_);
  }

  return output_tensor;

}

#endif //USE_OPENCL

INSTANTIATE_CLASS_CPU(ScaleOp_CPU);
INSTANTIATE_CLASS_GPU(ScaleOp_GPU);
// REGISTER_LAYER_CLASS(Scale);

}  // namespace hypertea
