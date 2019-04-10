#ifndef HYPERTEA_RELU_OP_HPP_
#define HYPERTEA_RELU_OP_HPP_

#include <vector>

#include "hypertea/operator.hpp"

namespace hypertea {



template <typename Dtype>
class ReLUOp_CPU {
 public:

  explicit ReLUOp_CPU(float negative_slope, bool inplace = false)
      : negative_slope_(negative_slope), inplace_(inplace) {}

  inline const char* type() const { return "ReLU"; }

  TensorCPU<Dtype> operator()(TensorCPU<Dtype> &input);


  private:

    float negative_slope_;
    bool inplace_;

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
  explicit ReLUOp_GPU(float negative_slope, bool inplace = false)
      : GPUFunctor<Dtype>(), negative_slope_(negative_slope), inplace_(inplace) {}

  virtual inline const char* type() const { return "ReLU"; }

  // virtual void Forward(const std::vector<cl_mem> bottom_datas,
  //     const std::vector<cl_mem> top_datas);

  virtual TensorGPU<Dtype> Forward(TensorGPU<Dtype> input_tensor);


  private:
    float negative_slope_;
    bool inplace_;

};

#endif //USE_OPENCL

}  // namespace hypertea

#endif  // HYPERTEA_RELU_OP_HPP_
