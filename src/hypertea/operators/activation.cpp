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


template<typename DeviceTensor>
DeviceTensor SoftMaxOp<DeviceTensor>::operator()(DeviceTensor input) {

	auto output = inplace_? input : DeviceTensor(input.count());

	inplace_exp(output);

	inplace_channeled_scal(
		output, 
		float(1) / channeled_sum(output, spatial_dim_), 
		input.count() / spatial_dim_,
		spatial_dim_
	);


	return output;

}
DEFINE_FORWARD_FUNC(SoftMaxOp);


}  // namespace hypertea
