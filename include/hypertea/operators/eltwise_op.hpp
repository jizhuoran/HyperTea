#ifndef HYPERTEA_ELTWISE_OP_HPP_
#define HYPERTEA_ELTWISE_OP_HPP_

#include <vector>

#include "hypertea/operator.hpp"

namespace hypertea {


enum EltwiseParameter_EltwiseOp {
  EltwiseParameter_EltwiseOp_PROD = 0,
  EltwiseParameter_EltwiseOp_SUM = 1,
  EltwiseParameter_EltwiseOp_MAX = 2
};


/**
 * @brief Compute elementwise operations, such as product and sum,
 *        along multiple input Blobs.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class EltwiseOp_CPU : public CPUFunctor<Dtype> {
 public:
  explicit EltwiseOp_CPU(int top_count, int bottom_nums, EltwiseParameter_EltwiseOp op, int* max_idx, std::vector<float> coeffs)
      : CPUFunctor<Dtype>(), top_count_(top_count), bottom_nums_(bottom_nums),
        op_(op), max_idx_(max_idx), coeffs_(coeffs) {}


  virtual inline const char* type() const { return "Eltwise"; }

 // protected:
  virtual void Forward(const std::vector<Dtype*> bottom_datas,
      const std::vector<Dtype*> top_datas);
  
  virtual TensorCPU<Dtype> Forward(TensorCPU<Dtype> &input) {}

private:
  
  int top_count_;
  int bottom_nums_;

  EltwiseParameter_EltwiseOp op_;
  int* max_idx_;
  // Dtype* coeffs_;
  std::vector<float> coeffs_;


};


#ifdef USE_OPENCL

template <typename Dtype>
class EltwiseOp_GPU : public GPUFunctor<Dtype> {
 public:

  explicit EltwiseOp_GPU(int top_count, int bottom_nums, EltwiseParameter_EltwiseOp op, cl_mem max_idx, std::vector<float> coeffs)
      : GPUFunctor<Dtype>(), top_count_(top_count), bottom_nums_(bottom_nums),
        op_(op), max_idx_(max_idx), coeffs_(coeffs) {
          
        }



  virtual inline const char* type() const { return "Eltwise"; }

 // protected:

  virtual void Forward(const std::vector<cl_mem> bottom_datas,
      const std::vector<cl_mem> top_datas);
  
private:
  
  int top_count_;
  int bottom_nums_;

  EltwiseParameter_EltwiseOp op_;
  cl_mem max_idx_;
  std::vector<float> coeffs_;


};

#endif //USE_OPENCL


}  // namespace hypertea

#endif  // HYPERTEA_ELTWISE_OP_HPP_
