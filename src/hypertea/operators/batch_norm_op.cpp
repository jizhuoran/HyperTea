// #include <algorithm>
// #include <vector>
// #include <math.h>

#include "hypertea/operators/batch_norm_op.hpp"
#include "hypertea/common.hpp"

namespace hypertea {

 
template<typename DeviceTensor>
DeviceTensor BatchNormOp<DeviceTensor>::operator()(DeviceTensor& input) {

  DeviceTensor output = inplace_? input : input.duplicate();

  if (!use_global_stats_) {
    mean_var(input, *mean_, *variance_, channels_, spatial_dim_, eps_);
  } else {
    *variance_ += eps_;
    inplace_sqrt(*variance_);
  }


  inplace_channeled_sub(output, *mean_, channels_, spatial_dim_);


  if(weight_ != nullptr) {
    auto weight_with_var = *weight_ / *variance_;
    if (bias_ != nullptr) {
      inplace_channeled_scaladd(output, weight_with_var, *bias_, channels_, spatial_dim_);
    } else {
      inplace_channeled_scal(output, weight_with_var, channels_, spatial_dim_);
    }
  } else {
    inplace_inv(*variance_, 0);
    inplace_channeled_scal(output, *variance_, channels_, spatial_dim_);
  }

  return output;

}

DEFINE_FORWARD_FUNC(BatchNormOp);

}  // namespace hypertea
