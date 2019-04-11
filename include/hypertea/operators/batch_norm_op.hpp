#ifndef HYPERTEA_BATCHNORM_OP_HPP_
#define HYPERTEA_BATCHNORM_OP_HPP_

// #include <vector>

// #include "hypertea/operator.hpp"

namespace hypertea {


template <typename DeviceTensor>
class BatchNormOp {

public:
  
  explicit BatchNormOp(
    int channels, int spatial_dim,
    float eps,
    DeviceTensor* mean, DeviceTensor* variance,
    DeviceTensor* weight, DeviceTensor* bias,
    bool inplace = false
  ) : channels_(channels), spatial_dim_(spatial_dim),
      eps_(eps),
      mean_(mean), variance_(variance),
      weight_(weight), bias_(bias),
      inplace_(inplace) {

    if(mean == nullptr) {
      use_global_stats_ = false;
      mean_ = new DeviceTensor(channels);
      variance_ = new DeviceTensor(channels);
    }

  }

  ~BatchNormOp() {

    if(!use_global_stats_) {
      delete mean_;
      delete variance_;
    }

  }


  inline const char* type() const { return "BatchNorm"; }

  DeviceTensor operator()(DeviceTensor &input);

private:

  int channels_;
  int spatial_dim_;
  float eps_;

  DeviceTensor* mean_;
  DeviceTensor* variance_;
  DeviceTensor* weight_;
  DeviceTensor* bias_;

  bool inplace_;
  bool use_global_stats_ = true;

};



}  // namespace hypertea

#endif  // HYPERTEA_BATCHNORM_OP_HPP_
