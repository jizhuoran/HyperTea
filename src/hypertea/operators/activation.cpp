#include "hypertea/common.hpp"
#include "hypertea/operators/activation.hpp"

namespace hypertea {


template<typename DeviceTensor>
DeviceTensor PReLUOp<DeviceTensor>::operator()(DeviceTensor input) {

	DeviceTensor output = inplace_? input : input.duplicate();

	inplace_prelu(output, *weight_, channels_, inner_dim_);

	return output;
}
DEFINE_FORWARD_FUNC(PReLUOp);

 

template<typename DeviceTensor>
DeviceTensor ReLUOp<DeviceTensor>::operator()(DeviceTensor input) {
	return inplace_? DeviceTensor(inplace_relu(input, negative_slope_)) : outplace_relu(input, negative_slope_);
}
DEFINE_FORWARD_FUNC(ReLUOp);


template<typename DeviceTensor>
DeviceTensor TanHOp<DeviceTensor>::operator()(DeviceTensor input) {
	return inplace_? DeviceTensor(inplace_tanh(input)) : outplace_tanh(input);
}
DEFINE_FORWARD_FUNC(TanHOp);


template<typename DeviceTensor>
DeviceTensor ELUOp<DeviceTensor>::operator()(DeviceTensor input) {
	return inplace_?DeviceTensor(inplace_elu(input, alpha_)) : outplace_elu(input, alpha_);
}
DEFINE_FORWARD_FUNC(ELUOp);


}  // namespace hypertea
