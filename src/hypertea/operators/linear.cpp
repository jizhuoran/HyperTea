#include "hypertea/common.hpp"
#include "hypertea/operators/linear_op.hpp"

namespace hypertea {

template<typename DeviceTensor>
DeviceTensor LinearOp<DeviceTensor>::operator()(DeviceTensor input) {

	auto batch_size = input.count() / in_features_;

	DeviceTensor output = DeviceTensor(batch_size * out_features_);

	if(bias_) {
		for(auto& x : output.chunked_tensors(batch_size)) {
			x.copy_data(*bias_);
		}
	}

	inplace_gemm(
	 	CblasNoTrans, CblasTrans,
	    batch_size, out_features_, in_features_, (float)1.,
	    input, *weight_, (float)1., output
	);


	return output;

}

DEFINE_FORWARD_FUNC(LinearOp);


}  // namespace hypertea
