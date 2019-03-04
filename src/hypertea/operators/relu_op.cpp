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

  const cl_mem input_data = input_tensor.immutable_data();
  TensorGPU<Dtype> output_tensor = inplace_? input_tensor : TensorGPU<Dtype>(input_tensor.count());
  cl_mem output_data = output_tensor.mutable_data();

  cl_int ret;
  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "ReLUForward", &ret);
  OPENCL_CHECK(ret);


  int data_count = input_tensor.count();

  // Set arguments for kernel

  Dtype negative_slope_gpu = this->to_dtype(negative_slope_);

  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_data));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output_data));
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&data_count));
  OPENCL_CHECK(clSetKernelArg(kernel, 3, this->gpu_dtype_size(), (void *)&negative_slope_gpu));  

  size_t global_size = HYPERTEA_GET_BLOCKS(data_count);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  

  return output_tensor;
}

#endif //USE_OPENCL

INSTANTIATE_CLASS_CPU(ReLUOp_CPU);
INSTANTIATE_CLASS_GPU(ReLUOp_GPU);

}  // namespace hypertea
