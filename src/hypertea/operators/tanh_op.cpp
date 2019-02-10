// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <vector>

#include "hypertea/operators/tanh_op.hpp"

namespace hypertea {

template <>
void TanHOp_CPU<float>::Forward(const float* bottom_data,
      float* top_data) {

  for (int i = 0; i < data_count_; ++i) {
    top_data[i] = tanh(bottom_data[i]);
  }
}


#ifdef USE_OPENCL

template <typename Dtype>
void TanHOp_GPU<Dtype>::Forward(const cl_mem bottom_data,
      cl_mem top_data) {

  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "TanHForward", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bottom_data));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&top_data));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&data_count_));  

  size_t global_size = HYPERTEA_GET_BLOCKS(data_count_);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  
  
}

#endif //USE_OPENCL


INSTANTIATE_CLASS_CPU(TanHOp_CPU);
INSTANTIATE_CLASS_GPU(TanHOp_GPU);

}  // namespace hypertea
