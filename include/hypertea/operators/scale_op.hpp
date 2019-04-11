#ifndef HYPERTEA_SCALE_OP_HPP_
#define HYPERTEA_SCALE_OP_HPP_

namespace hypertea {

template <typename DeviceTensor>
class ScaleOp {

public:
  explicit ScaleOp(
    DeviceTensor* weight, DeviceTensor* bias, 
    int channels, int spatial_dim)
  : weight_(weight), bias_(bias),
    channels_(channels), spatial_dim_(spatial_dim) {}

  inline const char* type() const { return "Scale"; }
  DeviceTensor operator()(DeviceTensor &input);

private:
  DeviceTensor* bias_;
  DeviceTensor* weight_;
  
  int channels_;
  int spatial_dim_;
  bool inplace_ = false;

};

}  // namespace hypertea

#endif  // HYPERTEA_SCALE_OP_HPP_
