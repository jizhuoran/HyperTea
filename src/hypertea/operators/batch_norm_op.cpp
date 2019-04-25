// #include <algorithm>
// #include <vector>
// #include <math.h>

#include "hypertea/operators/batch_norm_op.hpp"
#include "hypertea/common.hpp"

namespace hypertea {

 
template<typename DeviceTensor>
DeviceTensor BatchNormOp<DeviceTensor>::operator()(DeviceTensor input) {

  DeviceTensor output = inplace_? input : input.duplicate();

  DeviceTensor variance(channels_);

  if (!use_global_stats_) {
    mean_var(input, *mean_, variance, channels_, spatial_dim_, eps_);
  } else {

    variance.copy_data(*variance_);
    variance += eps_;
    inplace_sqrt(variance);
    
  }


  inplace_channeled_sub(output, *mean_, channels_, spatial_dim_);


  if(weight_ != nullptr) {
    auto weight_with_var = *weight_ / variance;
    if (bias_ != nullptr) {
      inplace_channeled_scaladd(output, weight_with_var, *bias_, channels_, spatial_dim_);
    } else {
      inplace_channeled_scal(output, weight_with_var, channels_, spatial_dim_);
    }
  } else {
    inplace_inv(variance);
    inplace_channeled_scal(output, variance, channels_, spatial_dim_);
  }

  return output;

}


DEFINE_FORWARD_FUNC(BatchNormOp);

}  // namespace hypertea
