#include <algorithm>
#include <vector>

#include "hypertea/operators/elu_op.hpp"

namespace hypertea {


template <>
void ELUOp_CPU<float>::Forward(const std::vector<float*> bottom_datas,
      const std::vector<float*> top_datas) {

  for (int i = 0; i < data_count_; ++i) {
    top_datas[0][i] = std::max(bottom_datas[0][i], float(0))
        + alpha_ * (exp(std::min(bottom_datas[0][i], float(0))) - float(1));
  }

}


template <>
std::vector<Tensor<float> *> ELUOp_CPU<float>::Forward(std::vector<Tensor<float> *> inputs) {

  float* input = inputs[0]->data();
  Tensor<float>* output_tensor = new Tensor<float>(inputs[0]->size());
  float* output = output_tensor->data();

  for (int i = 0; i < inputs[0]->size(); ++i) {
    output[i] = std::max(input[i], float(0))
        + alpha_ * (exp(std::min(input[i], float(0))) - float(1));
  }

  return {output_tensor};

}


template <>
Tensor<float> ELUOp_CPU<float>::Forward1(Tensor<float> &inputs) {
  
  float* input_data = inputs.data();
  Tensor<float> output(inputs.size());
  float* output_data = output.data();

  for (int i = 0; i < inputs.size(); ++i) {
    output_data[i] = std::max(input_data[i], float(0))
        + alpha_ * (exp(std::min(input_data[i], float(0))) - float(1));
  }

  return output;

}



#ifdef USE_OPENCL

template <typename Dtype>
void ELUOp_GPU<Dtype>::Forward(const std::vector<cl_mem> bottom_datas,
      const std::vector<cl_mem> top_datas) {

  cl_int ret;
  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "ELUForward", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel

  Dtype alpha_gpu = this->to_dtype(alpha_);

  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bottom_datas[0]));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&top_datas[0]));
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&data_count_));
  OPENCL_CHECK(clSetKernelArg(kernel, 3, this->gpu_dtype_size(), (void *)&alpha_gpu));  

  size_t global_size = HYPERTEA_GET_BLOCKS(data_count_);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  

}
#endif //USE_OPENCL



INSTANTIATE_CLASS_CPU(ELUOp_CPU);
INSTANTIATE_CLASS_GPU(ELUOp_GPU);
// REGISTER_LAYER_CLASS(ELU);

}  // namespace hypertea
