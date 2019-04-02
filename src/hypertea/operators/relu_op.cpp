#include <algorithm>
#include <vector>

#include "hypertea/operators/relu_op.hpp"


namespace hypertea {


template <>
TensorCPU<float> ReLUOp_CPU<float>::Forward(TensorCPU<float> &input_tensor) {
  
  const float* input_data = input_tensor.immutable_data();
  float* output_data = inplace_? input_tensor.mutable_data() : new float[input_tensor.size()];

  for (int i = 0; i < input_tensor.size(); ++i) {
    output_data[i] = std::max(input_data[i], float(0))
        + negative_slope_ * std::min(input_data[i], float(0));
  }

  return inplace_? input_tensor:TensorCPU<float>(output_data, input_tensor.size());  

}



#ifdef USE_OPENCL

template <typename Dtype>
TensorGPU<Dtype> ReLUOp_GPU<Dtype>::Forward(TensorGPU<Dtype> input_tensor){
// void ReLUOp_GPU<Dtype>::Forward(const std::vector<cl_mem> bottom_datas,
      // const std::vector<cl_mem> top_datas) {

  if(inplace_) {
   input_tensor.relu(negative_slope_);
   return input_tensor;
  } {
    return gpu_relu(input_tensor, negative_slope_);
  }
}

#endif //USE_OPENCL

INSTANTIATE_CLASS_CPU(ReLUOp_CPU);
INSTANTIATE_CLASS_GPU(ReLUOp_GPU);

}  // namespace hypertea
