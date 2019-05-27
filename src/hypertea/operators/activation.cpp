#include "hypertea/common.hpp"
#include "hypertea/operators/activation.hpp"

namespace hypertea {


template<typename DeviceTensor>
DeviceTensor PReLUOp<DeviceTensor>::operator()(DeviceTensor input) {

    std::cout << "DEBUG: PReLUOp 0" <<std::endl;


	DeviceTensor output = inplace_? input : input.duplicate();

    std::cout << "DEBUG: PReLUOp 1" <<std::endl;


	inplace_prelu(output, *weight_, channels_, inner_dim_);
	return output;
}
DEFINE_FORWARD_FUNC(PReLUOp);

 

template<typename DeviceTensor>
DeviceTensor ReLUOp<DeviceTensor>::operator()(DeviceTensor input) {
	auto output = inplace_? input : input.duplicate();
	inplace_relu(output, negative_slope_);
	return output;
}
DEFINE_FORWARD_FUNC(ReLUOp);


template<typename DeviceTensor>
DeviceTensor TanHOp<DeviceTensor>::operator()(DeviceTensor input) {
	auto output = inplace_? input : input.duplicate();
	inplace_tanh(output);
	return output;
}
DEFINE_FORWARD_FUNC(TanHOp);


template<typename DeviceTensor>
DeviceTensor ELUOp<DeviceTensor>::operator()(DeviceTensor input) {
	auto output = inplace_? input : input.duplicate();
	inplace_elu(output, alpha_);
	return output;
}
DEFINE_FORWARD_FUNC(ELUOp);


template<typename DeviceTensor>
DeviceTensor SoftMaxOp<DeviceTensor>::operator()(DeviceTensor input) {

	auto output = inplace_? input : input.duplicate();

	inplace_exp(output);

	auto channed_sum = channeled_sum(output, spatial_dim_);

	inplace_channeled_scal(
		output, 
		inplace_inv(channed_sum), 
		input.count() / spatial_dim_,
		spatial_dim_
	);


	return output;

}
DEFINE_FORWARD_FUNC(SoftMaxOp);


}  // namespace hypertea
