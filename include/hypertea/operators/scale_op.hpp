#ifndef HYPERTEA_SCALE_OP_HPP_
#define HYPERTEA_SCALE_OP_HPP_

#include "hypertea/operator.hpp"

namespace hypertea {

template <typename DeviceTensor>
class ScaleOp  : public TensorOperator<DeviceTensor>{

public:
  explicit ScaleOp(
    DeviceTensor* weight, DeviceTensor* bias, 
    int channels, int spatial_dim)
  : TensorOperator<DeviceTensor>(),
    weight_(weight), bias_(bias),
    channels_(channels), spatial_dim_(spatial_dim) {}

  virtual inline const char* type() const override { return "Scale"; }
  virtual DeviceTensor operator()(DeviceTensor &input) override;

private:
  DeviceTensor* bias_;
  DeviceTensor* weight_;
  
  int channels_;
  int spatial_dim_;
  bool inplace_ = false;

};

}  // namespace hypertea

#endif  // HYPERTEA_SCALE_OP_HPP_
