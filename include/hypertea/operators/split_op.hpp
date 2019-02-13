#ifndef HYPERTEA_SPLIT_OP_HPP_
#define HYPERTEA_SPLIT_OP_HPP_

#include <vector>

#include "hypertea/operator.hpp"

namespace hypertea {

/**
 * @brief Creates a "split" path in the network by copying the bottom Blob
 *        into multiple top Blob%s to be used by multiple consuming layers.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class SplitOp_CPU : public CPUFunctor<Dtype> {
 public:
  explicit SplitOp_CPU(int data_count)
      : CPUFunctor<Dtype>(), data_count_(data_count) {}

  virtual inline const char* type() const { return "Split"; }

 // protected:
  virtual void Forward(const std::vector<Dtype*> bottom_datas,
      const std::vector<Dtype*> top_datas);

  int data_count_;
};

#ifdef USE_OPENCL

template <typename Dtype>
class SplitOp_GPU : public GPUFunctor<Dtype> {
 public:
  explicit SplitOp_GPU(int data_count)
      : GPUFunctor<Dtype>(), data_count_(data_count) {}

  virtual inline const char* type() const { return "Split"; }

 // protected:
  virtual void Forward(const std::vector<cl_mem> bottom_datas,
      const std::vector<cl_mem> top_datas);

  int data_count_;
};

#endif //USE_OPENCL

}  // namespace hypertea

#endif  // HYPERTEA_SPLIT_OP_HPP_
