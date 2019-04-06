// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <vector>

#include "hypertea/operators/tanh_op.hpp"

namespace hypertea {


template <>
TensorCPU<float> TanHOp_CPU<float>::Forward(TensorCPU<float> &input_tensor) {
  
  return inplace_? TensorCPU<float>(input_tensor.tanh()) : cpu_tanh(input_tensor);
  // const float* input_data = input_tensor.immutable_data();
  // float* output_data = inplace_? input_tensor.mutable_data() : new float[input_tensor.size()];

  // for (int i = 0; i < input_tensor.size(); ++i) {
  //     output_data[i] = tanh(input_data[i]);
  // }

  // return inplace_? input_tensor:TensorCPU<float>(output_data, input_tensor.size());  

}



#ifdef USE_OPENCL

template <typename Dtype>
TensorGPU<Dtype> TanHOp_GPU<Dtype>::Forward(TensorGPU<Dtype> input_tensor){

  return inplace_? TensorGPU<Dtype>(input_tensor.tanh()) : gpu_tanh(input_tensor);

}

#endif //USE_OPENCL


INSTANTIATE_CLASS_CPU(TanHOp_CPU);
INSTANTIATE_CLASS_GPU(TanHOp_GPU);

}  // namespace hypertea
