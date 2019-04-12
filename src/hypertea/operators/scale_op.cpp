#include "hypertea/common.hpp"
#include "hypertea/operators/scale_op.hpp"

namespace hypertea {

template<typename DeviceTensor>
DeviceTensor ScaleOp<DeviceTensor>::operator()(DeviceTensor input) {

  DeviceTensor output = inplace_? input : input.duplicate();

  if (bias_ != nullptr) {
      inplace_channeled_scaladd(output, *weight_, *bias_, channels_, spatial_dim_);
  } else {
      inplace_channeled_scal(output, *weight_, channels_, spatial_dim_);
  }

  return output;
}

DEFINE_FORWARD_FUNC(ScaleOp);


}  // namespace hypertea
