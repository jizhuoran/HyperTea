
#ifndef HYPERTEA_TENSOR_OP_H_
#define HYPERTEA_TENSOR_OP_H_

#include "hypertea/tensor.hpp"

namespace hypertea {

template<typename Dtype> TensorGPU<Dtype> operator+ (const TensorGPU<Dtype>& lhs, const TensorGPU<Dtype>& rhs) {
	return gpu_add(lhs ,rhs);
}



} // namespace hypertea
#endif //HYPERTEA_TENSOR_OP_H_