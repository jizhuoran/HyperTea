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



  std::vector<std::pair<size_t, const void *> > arg_list {
    std::make_pair(sizeof(cl_mem), (void *)&input_data),
    std::make_pair(sizeof(cl_mem), (void *)&this->weight_data_),
    std::make_pair(sizeof(cl_mem), (void *)&output_data)
  };

  if (this->bias_.count() != 0) {
    arg_list.push_back(std::make_pair(sizeof(cl_mem), (void *)&this->bias_data_));
  }


  opencl_launch_wrapper(
    OpenCLHandler::Get().conv_program,
    this->kernel_name_,
    arg_list,
    this->global_size_,
    this->local_size_
  );
  
  return output_tensor;


}
#endif //USE_OPENCL


INSTANTIATE_CLASS_CPU(ConvolutionOp_CPU);
INSTANTIATE_CLASS_GPU(ConvolutionOp_GPU);

}  // namespace hypertea
