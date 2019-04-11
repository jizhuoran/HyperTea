#ifndef HYPERTEA_LIBDNNCONV_OP_HPP_
#define HYPERTEA_LIBDNNCONV_OP_HPP_

#include <vector>
#include "hypertea/operator.hpp"

namespace hypertea {


template <typename Dtype>
class LibDNNBase {

public:
  explicit LibDNNBase(
    std::string kernel_name,
    int top_count,
    TensorGPU<float>* weight, 
    TensorGPU<float>* bias,
    std::vector<int> local,
    std::vector<int> global)

    :kernel_name_(kernel_name),
    top_count_(top_count), 
    weight_(weight), bias_(bias) {

    local_size_.push_back(local[0]);
    local_size_.push_back(local[1]);
    local_size_.push_back(local[2]);

    global_size_.push_back(global[0]);
    global_size_.push_back(global[1]);
    global_size_.push_back(global[2]);

  }


protected:

  int top_count_;
  std::string kernel_name_;

  TensorGPU<float>* weight_;
  TensorGPU<float>* bias_;


  std::vector<size_t> local_size_;
  std::vector<size_t> global_size_;

};


#ifdef USE_OPENCL

template <typename Dtype>
class LibDNNConvOp : public LibDNNBase<Dtype> {
 public:

  explicit LibDNNConvOp(
    std::string kernel_name,
    int top_count,
    TensorGPU<float>* weight, 
    TensorGPU<float>* bias,
    std::vector<int> local,
    std::vector<int> global)
    : LibDNNBase<Dtype>(
      kernel_name, top_count, weight, bias,
      local, global) { }

  inline const char* type() const { return "Convolution"; }

  
  TensorGPU<Dtype> operator()(TensorGPU<Dtype> &input);
  
};


template <typename Dtype>
class LibDNNDeconvOp : public LibDNNBase<Dtype> {
 public:

  explicit LibDNNDeconvOp(
    std::string kernel_name,
    int top_count,
    TensorGPU<float>* weight, 
    TensorGPU<float>* bias,
    std::vector<int> local,
    std::vector<int> global)
    : LibDNNBase<Dtype>(
      kernel_name, top_count, weight, bias,
      local, global) { }

  inline const char* type() const { return "Deconvolution"; }

  TensorGPU<Dtype> operator()(TensorGPU<Dtype> &input);

  
};


#endif //USE_OPENCL


}  // namespace hypertea

#endif  // HYPERTEA_LIBDNNCONV_OP_HPP_
