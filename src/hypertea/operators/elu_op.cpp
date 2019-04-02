#include <algorithm>
#include <vector>

#include "hypertea/operators/elu_op.hpp"

namespace hypertea {


template <>
TensorCPU<float> ELUOp_CPU<float>::Forward(TensorCPU<float> &input_tensor) {
  
  const float* input_data = input_tensor.immutable_data();
  float* output_data = inplace_? input_tensor.mutable_data() : new float[input_tensor.size()];

  for (int i = 0; i < input_tensor.size(); ++i) {
      output_data[i] = std::max(input_data[i], float(0))
          + alpha_ * (exp(std::min(input_data[i], float(0))) - float(1));
  }

  return inplace_? input_tensor:TensorCPU<float>(output_data, input_tensor.size());  

}



#ifdef USE_OPENCL

template <typename Dtype>
TensorGPU<Dtype> ELUOp_GPU<Dtype>::Forward(TensorGPU<Dtype> input_tensor){

  if(inplace_) {
   input_tensor.elu(alpha_);
   return input_tensor;
  } {
    return gpu_elu(input_tensor, alpha_);
  }

}
#endif //USE_OPENCL



INSTANTIATE_CLASS_CPU(ELUOp_CPU);
INSTANTIATE_CLASS_GPU(ELUOp_GPU);
// REGISTER_LAYER_CLASS(ELU);

}  // namespace hypertea
