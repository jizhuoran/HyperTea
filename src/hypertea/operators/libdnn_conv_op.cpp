#include <vector>

#include "hypertea/operators/libdnn_conv_op.hpp"

namespace hypertea {


#ifdef USE_OPENCL

template <typename Dtype>
TensorGPU<Dtype> LibDNNConvOp<Dtype>::operator()(TensorGPU<Dtype>& input) {

  const cl_mem input_data = input.immutable_data();
  TensorGPU<Dtype> output(this->top_count_);
  cl_mem output_data = output.mutable_data();

  auto weight_data_ = this->weight_->immutable_data();
  std::vector<std::pair<size_t, const void *> > arg_list {
    std::make_pair(sizeof(cl_mem), (void *)&input_data),
    std::make_pair(sizeof(cl_mem), (void *)&weight_data_),
    std::make_pair(sizeof(cl_mem), (void *)&output_data)
  };

  if (this->bias_ != nullptr) {
    auto bias_data_ = this->bias_->immutable_data();
    arg_list.push_back(std::make_pair(sizeof(cl_mem), (void *)&bias_data_));
  }


  opencl_launch_wrapper(
    OpenCLHandler::Get().conv_program,
    this->kernel_name_,
    arg_list,
    this->global_size_,
    this->local_size_
  );
  
  return output;

}

template TensorGPU<float> LibDNNConvOp<float>::operator()(TensorGPU<float>& input);
template TensorGPU<half> LibDNNConvOp<half>::operator()(TensorGPU<half>& input);


template <typename Dtype>
TensorGPU<Dtype> LibDNNDeconvOp<Dtype>::operator()(TensorGPU<Dtype>& input) {

  const cl_mem input_data = input.immutable_data();
  TensorGPU<Dtype> output(this->top_count_);
  cl_mem output_data = output.mutable_data();

  auto weight_data_ = this->weight_->immutable_data();
  std::vector<std::pair<size_t, const void *> > arg_list {
    std::make_pair(sizeof(cl_mem), (void *)&input_data),
    std::make_pair(sizeof(cl_mem), (void *)&weight_data_),
    std::make_pair(sizeof(cl_mem), (void *)&output_data)
  };

  if (this->bias_ != nullptr) {
    auto bias_data_ = this->bias_->immutable_data();
    arg_list.push_back(std::make_pair(sizeof(cl_mem), (void *)&bias_data_));
  }


  opencl_launch_wrapper(
    OpenCLHandler::Get().conv_program,
    this->kernel_name_,
    arg_list,
    this->global_size_,
    this->local_size_
  );
  
  return output;

}


template TensorGPU<float> LibDNNDeconvOp<float>::operator()(TensorGPU<float>& input);
template TensorGPU<half> LibDNNDeconvOp<half>::operator()(TensorGPU<half>& input);


#endif //USE_OPENCL


}  // namespace hypertea
