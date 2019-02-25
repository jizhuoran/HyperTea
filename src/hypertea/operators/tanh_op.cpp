// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <vector>

#include "hypertea/operators/tanh_op.hpp"

namespace hypertea {

// template <>
// void TanHOp_CPU<float>::Forward(const std::vector<float*> bottom_datas,
//       const std::vector<float*> top_datas) {

//   for (int i = 0; i < data_count_; ++i) {
//     top_datas[0][i] = tanh(bottom_datas[0][i]);
//   }
// }



template <>
std::vector<Tensor<float> *> TanHOp_CPU<float>::Forward(std::vector<Tensor<float> *> inputs) {

  float* input = inputs[0]->data();
  Tensor<float>* output_tensor = new Tensor<float>(inputs[0]->size());
  float* output = output_tensor->data();

  for (int i = 0; i < inputs[0]->size(); ++i) {
    output[i] = tanh(input[i]);
  }

  return {output_tensor};

}



#ifdef USE_OPENCL

template <typename Dtype>
void TanHOp_GPU<Dtype>::Forward(const std::vector<cl_mem> bottom_datas,
      const std::vector<cl_mem> top_datas) {

  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "TanHForward", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bottom_datas[0]));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&top_datas[0]));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&data_count_));  

  size_t global_size = HYPERTEA_GET_BLOCKS(data_count_);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  
  
}

#endif //USE_OPENCL


INSTANTIATE_CLASS_CPU(TanHOp_CPU);
INSTANTIATE_CLASS_GPU(TanHOp_GPU);

}  // namespace hypertea
