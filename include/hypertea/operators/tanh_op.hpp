#ifndef HYPERTEA_TANH_OP_HPP_
#define HYPERTEA_TANH_OP_HPP_

#include <vector>

#include "hypertea/operator.hpp"

namespace hypertea {

/**
 * @brief TanH hyperbolic tangent non-linearity @f$
 *         y = \frac{\exp(2x) - 1}{\exp(2x) + 1}
 *     @f$, popular in auto-encoders.
 *
 * Note that the gradient vanishes as the values move away from 0.
 * The ReLULayer is often a better choice for this reason.
 */
template <typename Dtype>
class TanHOp_CPU : public CPUFunctor<Dtype> {
 public:
  explicit TanHOp_CPU(int data_count)
      : CPUFunctor<Dtype>(), data_count_(data_count) {}

  virtual inline const char* type() const { return "TanH"; }

  virtual void Forward(const Dtype* bottom_data,
      Dtype* top_data);

private:
  int data_count_;

};

#ifdef USE_OPENCL

template <typename Dtype>
class TanHOp_GPU : public GPUFunctor<Dtype> {
 public:
  explicit TanHOp_GPU(int data_count)
      : GPUFunctor<Dtype>(), data_count_(data_count) {}

  virtual inline const char* type() const { return "TanH"; }


  virtual void Forward(const cl_mem bottom_data,
      cl_mem top_data);

private:
  int data_count_;

};

#endif //USE_OPENCL


}  // namespace hypertea

#endif  // HYPERTEA_TANH_OP_HPP_
