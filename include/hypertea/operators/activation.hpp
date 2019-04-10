#ifndef HYPERTEA_ACTIVATION_OP_HPP_
#define HYPERTEA_ACTIVATION_OP_HPP_

#include <vector>

#include "hypertea/operator.hpp"

namespace hypertea {


template <typename DeviceTensor>
class ReLUOp {
 public:

  explicit ReLUOp(float negative_slope, bool inplace = false)
      : negative_slope_(negative_slope), inplace_(inplace) {}

  inline const char* type() const { return "ReLU"; }

  DeviceTensor operator()(DeviceTensor &input);

  private:

    float negative_slope_;
    bool inplace_;

};


template <typename DeviceTensor>
class TanHOp {
 public:
  explicit TanHOp(bool inplace = false)
      : inplace_(inplace) {}

  inline const char* type() const { return "TanH"; }

  DeviceTensor operator()(DeviceTensor &input);

private:
  bool inplace_;

};


template <typename DeviceTensor>
class ELUOp {
 public:
  

  explicit ELUOp(float alpha, bool inplace = false)
      : alpha_(alpha), inplace_(inplace) {}

  inline const char* type() const { return "ELU"; }

  DeviceTensor operator()(DeviceTensor &input);

  private:
    float alpha_;
    bool inplace_;


};



}  // namespace hypertea

#endif  // HYPERTEA_ACTIVATION_OP_HPP_
