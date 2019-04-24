#include "hypertea/common.hpp"
#include "hypertea/operators/linear_op.hpp"

namespace hypertea {

template<typename DeviceTensor>
DeviceTensor LinearOp<DeviceTensor>::operator()(DeviceTensor input) {

	auto batch_size = input.count() / in_features_;

	DeviceTensor output = DeviceTensor(batch_size * out_features_, 0);

	if(bias_ != nullptr) {
		for(auto& x : output.chunked_tensors(batch_size)) {
			x.copy_data(*bias_);
		}
	}

	inplace_gemm(
	 	CblasNoTrans, CblasNoTrans,
	    batch_size, out_features_, in_features_, (float)1.,
	    input, *weight_, (float)1., output
	);


	return output;

}

DEFINE_FORWARD_FUNC(LinearOp);





template<typename DeviceTensor>
DeviceTensor EmbeddingOp<DeviceTensor>::operator()(std::vector<int> input) {


	DeviceTensor output = DeviceTensor(input.size() * embedding_dim_);

	auto embedding_weight = (cl_mem) weight_->immutable_data();
	auto output_data = (cl_mem) output.mutable_data();
	

	// std::cout << "The embedding dim is " << embedding_dim_ << std::endl;
	// std::cout << "The size dim is " << output.type_size() << std::endl;
	// std::cout << "The input size is " << input.size() << std::endl;


	for (int i = 0; i < input.size(); ++i) {
		
		// std::cout << "we are going to copy " << input[i] << "'s input" << std::endl;

		OPENCL_CHECK(clEnqueueCopyBuffer(OpenCLHandler::Get().commandQueue, 
    		embedding_weight, 
    		output_data, 
    		output.type_size() * embedding_dim_ * input[i], output.type_size() * embedding_dim_ * i, 
    		output.type_size() * embedding_dim_, 0, NULL, NULL)
		);
	}

	// exit(0);

	return output;

}

template TensorCPU<float> EmbeddingOp<TensorCPU<float>>::operator()(std::vector<int> input);
template TensorGPU<float> EmbeddingOp<TensorGPU<float>>::operator()(std::vector<int> input);
template TensorGPU<half> EmbeddingOp<TensorGPU<half>>::operator()(std::vector<int> input);


}  // namespace hypertea
