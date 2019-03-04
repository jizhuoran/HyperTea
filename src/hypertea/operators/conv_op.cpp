#include <vector>

#include "hypertea/operators/conv_op.hpp"

namespace hypertea {

template <>
void ConvolutionOp_CPU<float>::Forward(const std::vector<float*> bottom_datas,
      const std::vector<float*> top_datas) {

  for (int i = 0; i < bottom_datas.size(); ++i) {
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_datas[i] + n * this->bottom_dim_, weight_,
          top_datas[i] + n * this->top_dim_);
      if (this->bias_) {
        this->forward_cpu_bias(top_datas[i] + n * this->top_dim_, bias_);
      }
    }
  }

}



template <>
TensorCPU<float> ConvolutionOp_CPU<float>::Forward(TensorCPU<float> &input_tensor) {

  const float* input_data = input_tensor.immutable_data();
  float* output_data = new float[this->top_count_];

  for (int n = 0; n < this->num_; ++n) {
    this->forward_cpu_gemm(input_data + n * this->bottom_dim_, weight_,
        output_data + n * this->top_dim_);
    if (this->bias_) {
      this->forward_cpu_bias(output_data + n * this->top_dim_, bias_);
    }
  }

  return TensorCPU<float>(output_data, top_count_);
}

#ifdef USE_OPENCL

template <typename Dtype>
TensorGPU<Dtype> ConvolutionOp_GPU<Dtype>::Forward(TensorGPU<Dtype> input_tensor){

  
  const cl_mem input_data = input_tensor.immutable_data();
  TensorGPU<Dtype> output_tensor(this->top_count_);
  cl_mem output_data = output_tensor.mutable_data();


  cl_int ret = -1;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().conv_program, this->kernel_name_.c_str(), &ret);
  OPENCL_CHECK(ret);

  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_data));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&this->weight_));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&output_data));

  if (this->bias_) {
    OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&this->bias_));
  }

  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 3, NULL, this->global_size_, this->local_size_, 0, NULL, NULL));  


  // std::cout << "come to here" << std::endl;

  return output_tensor;
}
#endif //USE_OPENCL


INSTANTIATE_CLASS_CPU(ConvolutionOp_CPU);
INSTANTIATE_CLASS_GPU(ConvolutionOp_GPU);

}  // namespace hypertea
