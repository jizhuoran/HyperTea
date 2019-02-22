#include "hypertea/tensor.hpp"


namespace hypertea {


template <typename Dtype>
Tensor<Dtype> Tensor<Dtype>::add(Tensor<Dtype> & other, Dtype alpha) {

	Tensor<Dtype> result(this->size());

	memcpy(result.data(), this->data(), this->size() * sizeof(Dtype));

  	hypertea_axpy(this->size(), alpha, other.data(), result.data());

  	return result;
}
template Tensor<float> Tensor<float>::add(Tensor<float> &other, float alpha = 1);



template <typename Dtype>
Tensor<Dtype>& Tensor<Dtype>::operator+=(const Tensor<Dtype> & other) {
	hypertea_add<Dtype>(this->count(), this->immutable_data(), other.immutable_data(), this->data());
  	return *this;
}
template Tensor<float>& Tensor<float>::operator+=(const Tensor<float> & other);






template <typename Dtype>
Tensor<Dtype> operator+(const Tensor<Dtype>& lhs, const Tensor<Dtype>& rhs) {
    
    Tensor<Dtype> result(lhs.count());

    hypertea_add<Dtype>(lhs.count(), lhs.immutable_data(), rhs.immutable_data(), result.data());

	return result;
}

template Tensor<float> operator+(const Tensor<float>& a, const Tensor<float>& rhs);


template <typename Dtype>
Tensor<Dtype> operator-(const Tensor<Dtype>& lhs, const Tensor<Dtype>& rhs) {
    
    Tensor<Dtype> result(lhs.count());

    hypertea_sub<Dtype>(lhs.count(), lhs.immutable_data(), rhs.immutable_data(), result.data());

	return result;
}

template Tensor<float> operator-(const Tensor<float>& a, const Tensor<float>& rhs);



	
} //namespace hypertea