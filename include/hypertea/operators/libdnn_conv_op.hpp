#ifndef HYPERTEA_LIBDNNCONV_OP_HPP_
#define HYPERTEA_LIBDNNCONV_OP_HPP_

#include <vector>
#include "hypertea/operator.hpp"

namespace hypertea {


template <typename DeviceTensor>
class LibDNNBase  : public TensorOperator<DeviceTensor>{

public:
  explicit LibDNNBase(
    std::string kernel_name,
    int top_count,
    DeviceTensor* weight, 
    DeviceTensor* bias,
    std::vector<size_t> local,
    std::vector<size_t> global)

    :kernel_name_(kernel_name),
    top_count_(top_count), 
    weight_(weight), bias_(bias),
    local_size_(local), global_size_(global) {

  }


protected:

  int top_count_;
  std::string kernel_name_;

  DeviceTensor* weight_;
  DeviceTensor* bias_;


  std::vector<size_t> local_size_;
  std::vector<size_t> global_size_;

};


#ifdef USE_OPENCL

template <typename DeviceTensor>
class LibDNNConvOp : public LibDNNBase<DeviceTensor> {
 public:

  explicit LibDNNConvOp(
    std::string kernel_name,
    int top_count,
    DeviceTensor* weight, 
    DeviceTensor* bias,
    std::vector<size_t> local,
    std::vector<size_t> global)
    : LibDNNBase<DeviceTensor>(
      kernel_name, top_count, weight, bias,
      local, global) { }

  virtual inline const char* type() const override { return "Convolution"; }

  
  virtual DeviceTensor operator()(DeviceTensor input) override;
  
};


template <typename DeviceTensor>
class LibDNNDeconvOp : public LibDNNBase<DeviceTensor> {
 public:

  explicit LibDNNDeconvOp(
    std::string kernel_name,
    int top_count,
    DeviceTensor* weight, 
    DeviceTensor* bias,
    std::vector<size_t> local,
    std::vector<size_t> global)
    : LibDNNBase<DeviceTensor>(
      kernel_name, top_count, weight, bias,
      local, global) { }

  virtual inline const char* type() const override { return "Deconvolution"; }

  virtual DeviceTensor operator()(DeviceTensor input) override;

  
};


#endif //USE_OPENCL


}  // namespace hypertea

#endif  // HYPERTEA_LIBDNNCONV_OP_HPP_
