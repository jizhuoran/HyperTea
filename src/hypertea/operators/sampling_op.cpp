#include "hypertea/common.hpp"
#include "hypertea/operators/sampling_op.hpp"

namespace hypertea {


template<typename DeviceTensor>
DeviceTensor UpSampling2D<DeviceTensor>::operator()(DeviceTensor input) {
	return upsampling_2d(input, scale_, height_, width_, height_* width_);
}
DEFINE_FORWARD_FUNC(UpSampling2D);

 

}  // namespace hypertea
