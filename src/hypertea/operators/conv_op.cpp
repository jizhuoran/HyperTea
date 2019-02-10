#include <vector>

#include "hypertea/operators/conv_op.hpp"

namespace hypertea {



template <>
void ConvolutionOp_CPU<float>::Forward(const std::vector<float*> bottom_datas,
      const std::vector<float*> top_datas) {

  for (int i = 0; i < bottom_size_; ++i) {
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_datas[i] + n * this->bottom_dim_, weight_,
          top_datas[i] + n * this->top_dim_);
      if (this->bias_) {
        this->forward_cpu_bias(top_datas[i] + n * this->top_dim_, bias_);
      }
    }
  }

}

#ifdef USE_OPENCL

template <typename Dtype>
void ConvolutionOp_GPU<Dtype>::Forward(const std::vector<cl_mem> bottom_datas,
      const std::vector<cl_mem> top_datas) {

  
  cl_int ret = -1;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().conv_program, this->kernel_name_.c_str(), &ret);

  for (int i = 0; i < this->bottom_size_; ++i) {

    OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bottom_datas[i]));  
    OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&this->weight_));  
    OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&top_datas[i]));

    if (this->bias_) {
      OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&this->bias_));
    }

    OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 3, NULL, this->global_size_, this->local_size_, 0, NULL, NULL));  

  }

}
#endif //USE_OPENCL


INSTANTIATE_CLASS_CPU(ConvolutionOp_CPU);
INSTANTIATE_CLASS_GPU(ConvolutionOp_GPU);

}  // namespace hypertea
