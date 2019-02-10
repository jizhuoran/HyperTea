#include <algorithm>
#include <vector>

#include "hypertea/operators/relu_op.hpp"


namespace hypertea {

template <>
void ReLUOp_CPU<float>::Forward(const float* bottom_data,
      float* top_data) {

  for (int i = 0; i < data_count_; ++i) {
    top_data[i] = std::max(bottom_data[i], float(0))
        + negative_slope_ * std::min(bottom_data[i], float(0));
  }

}


#ifdef USE_OPENCL

template <typename Dtype>
void ReLUOp_GPU<Dtype>::Forward(const cl_mem bottom_data,
      cl_mem top_data) {


  cl_int ret;
  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "ReLUForward", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel

  Dtype negative_slope_gpu = this->to_dtype(negative_slope_);

  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bottom_data));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&top_data));
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&data_count_));
  OPENCL_CHECK(clSetKernelArg(kernel, 3, this->gpu_dtype_size(), (void *)&negative_slope_gpu));  

  size_t global_size = HYPERTEA_GET_BLOCKS(data_count_);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  

}

#endif //USE_OPENCL

INSTANTIATE_CLASS_CPU(ReLUOp_CPU);
INSTANTIATE_CLASS_GPU(ReLUOp_GPU);

}  // namespace hypertea
