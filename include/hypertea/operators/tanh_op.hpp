#ifndef HYPERTEA_TANH_OP_HPP_
#define HYPERTEA_TANH_OP_HPP_

#include <vector>

#include "hypertea/operator.hpp"

namespace hypertea {


template <typename Dtype>
class TanHOp_CPU : public CPUFunctor<Dtype> {
 public:
  explicit TanHOp_CPU(bool inplace = false)
      : CPUFunctor<Dtype>(), inplace_(inplace) {}

  virtual inline const char* type() const { return "TanH"; }

  // virtual void Forward(const std::vector<Dtype*> bottom_datas,
      // const std::vector<Dtype*> top_datas);
  virtual TensorCPU<Dtype> Forward(TensorCPU<Dtype> &input_tensor);
  
  // virtual std::vector<Tensor<Dtype> *> Forward(const std::vector<Tensor<Dtype> *> inputs);

private:
  bool inplace_;

  // int data_count_;

};

#ifdef USE_OPENCL

template <typename Dtype>
class TanHOp_GPU : public GPUFunctor<Dtype> {
 public:
  explicit TanHOp_GPU(bool inplace_ = false)
      : GPUFunctor<Dtype>(), inplace_(inplace_) {}

  virtual inline const char* type() const { return "TanH"; }


  // virtual void Forward(const std::vector<cl_mem> bottom_datas,
      // const std::vector<cl_mem> top_datas);


  virtual TensorGPU<Dtype> Forward(TensorGPU<Dtype> input_tensor);

private:
  bool inplace_;

};

#endif //USE_OPENCL


}  // namespace hypertea

#endif  // HYPERTEA_TANH_OP_HPP_
