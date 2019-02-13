#ifndef HYPERTEA_ELU_OP_HPP_
#define HYPERTEA_ELU_OP_HPP_

#include <vector>

#include "hypertea/operator.hpp"


namespace hypertea {

/**
 * @brief Exponential Linear Unit non-linearity @f$
 *        y = \left\{
 *        \begin{array}{lr}
 *            x                  & \mathrm{if} \; x > 0 \\
 *            \alpha (\exp(x)-1) & \mathrm{if} \; x \le 0
 *        \end{array} \right.
 *      @f$.  
 */
template <typename Dtype>
class ELUOp_CPU : public CPUFunctor<Dtype> {
 public:
  

  explicit ELUOp_CPU(int data_count, float alpha)
      : CPUFunctor<Dtype>(), data_count_(data_count), alpha_(alpha) {}

  virtual inline const char* type() const { return "ELU"; }

  virtual void Forward(const std::vector<Dtype*> bottom_datas,
      const std::vector<Dtype*> top_datas);



  private:
    int data_count_;
    float alpha_;



};

#ifdef USE_OPENCL

template <typename Dtype>
class ELUOp_GPU : public GPUFunctor<Dtype> {
 public:
  

  explicit ELUOp_GPU(int data_count, float alpha)
      : GPUFunctor<Dtype>(), data_count_(data_count), alpha_(alpha) {}

  virtual inline const char* type() const { return "ELU"; }

  virtual void Forward(const std::vector<cl_mem> bottom_datas,
      const std::vector<cl_mem> top_datas);


  private:
    int data_count_;
    float alpha_;



};

#endif //USE_OPENCL

}  // namespace hypertea

#endif  // HYPERTEA_ELU_OP_HPP_
