#include <algorithm>
#include <vector>

#include "hypertea/operators/relu_op.hpp"


namespace hypertea {


template <>
TensorCPU<float> ReLUOp_CPU<float>::operator()(TensorCPU<float> &input) {
  
  const float* input_data = input.immutable_data();
  float* output_data = inplace_? input.mutable_data() : new float[input.size()];

  for (int i = 0; i < input.size(); ++i) {
    output_data[i] = std::max(input_data[i], float(0))
        + negative_slope_ * std::min(input_data[i], float(0));
  }

  return inplace_? input:TensorCPU<float>(output_data, input.size());  

}



#ifdef USE_OPENCL

template <typename Dtype>
TensorGPU<Dtype> ReLUOp_GPU<Dtype>::Forward(TensorGPU<Dtype> input_tensor){

  return inplace_? TensorGPU<Dtype>(input_tensor.relu(negative_slope_)) : gpu_relu(input_tensor, negative_slope_);

}

#endif //USE_OPENCL

// INSTANTIATE_CLASS_CPU(ReLUOp_CPU);
INSTANTIATE_CLASS_GPU(ReLUOp_GPU);

}  // namespace hypertea
