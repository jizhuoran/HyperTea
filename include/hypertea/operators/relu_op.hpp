#ifndef HYPERTEA_RELU_OP_HPP_
#define HYPERTEA_RELU_OP_HPP_

#include <vector>

#include "hypertea/operator.hpp"

namespace hypertea {

/**
 * @brief Rectified Linear Unit non-linearity @f$ y = \max(0, x) @f$.
 *        The simple max is fast to compute, and the function does not saturate.
 */
template <typename Dtype>
class ReLUOp_CPU : public CPUFunctor<Dtype> {
 public:
  /**
   * @param param provides ReLUParameter relu_param,
   *     with ReLULayer options:
   *   - negative_slope (\b optional, default 0).
   *     the value @f$ \nu @f$ by which negative values are multiplied.
   */
  explicit ReLUOp_CPU(int data_count, float negative_slope)
      : CPUFunctor<Dtype>(), data_count_(data_count), negative_slope_(negative_slope) {}

  virtual inline const char* type() const { return "ReLU"; }

  virtual void Forward(const Dtype* bottom_data,
      Dtype* top_data);


  private:
    int data_count_;
    float negative_slope_;

};

#ifdef USE_OPENCL

template <typename Dtype>
class ReLUOp_GPU : public GPUFunctor<Dtype> {
 public:
  /**
   * @param param provides ReLUParameter relu_param,
   *     with ReLULayer options:
   *   - negative_slope (\b optional, default 0).
   *     the value @f$ \nu @f$ by which negative values are multiplied.
   */
  explicit ReLUOp_GPU(int data_count, float negative_slope)
      : GPUFunctor<Dtype>(), data_count_(data_count), negative_slope_(negative_slope) {}

  virtual inline const char* type() const { return "ReLU"; }

  virtual void Forward(const cl_mem bottom_data,
      cl_mem top_data);


  private:
    int data_count_;
    float negative_slope_;

};

#endif //USE_OPENCL

}  // namespace hypertea

#endif  // HYPERTEA_RELU_OP_HPP_
