#include "hypertea/tensor.hpp"


namespace hypertea {




template <typename Dtype>
TensorCPU<Dtype> TensorCPU<Dtype>::add(TensorCPU<Dtype> & other, Dtype alpha) {

	TensorCPU<Dtype> result(this->size());

	memcpy(result.mutable_data(), this->immutable_data(), this->size() * sizeof(Dtype));

  	hypertea_axpy(this->size(), alpha, other.immutable_data(), result.mutable_data());

  	return result;
}
template TensorCPU<float> TensorCPU<float>::add(TensorCPU<float> &other, float alpha = 1);



template <typename Dtype>
TensorCPU<Dtype>& TensorCPU<Dtype>::operator+=(const TensorCPU<Dtype> & other) {
	hypertea_add<Dtype>(this->count(), this->immutable_data(), other.immutable_data(), this->mutable_data());
  	return *this;
}
template TensorCPU<float>& TensorCPU<float>::operator+=(const TensorCPU<float> & other);


template <typename Dtype>
TensorCPU<Dtype>& TensorCPU<Dtype>::operator+=(const Dtype other) {
  hypertea_add_scalar<Dtype>(this->count(), other, this->mutable_data());
    return *this;
}
template TensorCPU<float>& TensorCPU<float>::operator+=(const float other);




template <typename Dtype>
TensorCPU<Dtype>& TensorCPU<Dtype>::operator*=(const Dtype other) {
  hypertea_scal<Dtype>(this->count(), other, this->mutable_data());
    return *this;
}
template TensorCPU<float>& TensorCPU<float>::operator*=(const float other);






template <typename Dtype>
TensorCPU<Dtype> operator+(const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs) {
    
    TensorCPU<Dtype> result(lhs.count());

    hypertea_add<Dtype>(lhs.count(), lhs.immutable_data(), rhs.immutable_data(), result.mutable_data());

	return result;
}
template TensorCPU<float> operator+(const TensorCPU<float>& a, const TensorCPU<float>& rhs);


template <typename Dtype>
TensorCPU<Dtype> operator+(const TensorCPU<Dtype>& lhs, const float rhs) {
    
    TensorCPU<Dtype> result(lhs.count());
    hypertea_copy(lhs.count(), lhs.immutable_data(), result.mutable_data());
    hypertea_add_scalar<Dtype>(lhs.count(), rhs, result.mutable_data());
  return result;
}
template TensorCPU<float> operator+(const TensorCPU<float>& a, const float rhs);




template <typename Dtype>
TensorCPU<Dtype> operator-(const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs) {
    
    TensorCPU<Dtype> result(lhs.count());

    hypertea_sub<Dtype>(lhs.count(), lhs.immutable_data(), rhs.immutable_data(), result.mutable_data());

	return result;
}
template TensorCPU<float> operator-(const TensorCPU<float>& a, const TensorCPU<float>& rhs);


template <typename Dtype>
TensorCPU<Dtype> operator-(const TensorCPU<Dtype>& lhs, const float rhs) {
    
    TensorCPU<Dtype> result(lhs.count());
    hypertea_copy(lhs.count(), lhs.immutable_data(), result.mutable_data());
    hypertea_add_scalar<Dtype>(lhs.count(), -rhs, result.mutable_data());
  return result;
}
template TensorCPU<float> operator-(const TensorCPU<float>& a, const float rhs);




template <typename Dtype>
TensorCPU<Dtype> operator*(const TensorCPU<Dtype>& lhs, float rhs) {
    
    TensorCPU<Dtype> result(lhs.count());
    hypertea_cpu_scale<Dtype>(lhs.count(), rhs, lhs.immutable_data(), result.mutable_data());
  return result;
}
template TensorCPU<float> operator*(const TensorCPU<float>& a, const float rhs);






















template <typename Dtype>
TensorGPU<Dtype>& TensorGPU<Dtype>::operator+=(const TensorGPU<Dtype> & other) {
  hypertea_gpu_add<Dtype>(this->count(), this->immutable_data(), other.immutable_data(), this->mutable_data());
    return *this;
}
template TensorGPU<float>& TensorGPU<float>::operator+=(const TensorGPU<float> & other);


template <typename Dtype>
TensorGPU<Dtype>& TensorGPU<Dtype>::operator+=(const Dtype other) {
  hypertea_gpu_add_scalar<Dtype>(this->count(), other, this->mutable_data());
    return *this;
}
template TensorGPU<float>& TensorGPU<float>::operator+=(const float other);




template <typename Dtype>
TensorGPU<Dtype>& TensorGPU<Dtype>::operator*=(const Dtype other) {
  hypertea_gpu_scal<Dtype>(this->count(), other, this->mutable_data());
    return *this;
}
template TensorGPU<float>& TensorGPU<float>::operator*=(const float other);


	
template <typename Dtype>
TensorGPU<Dtype> operator+(const TensorGPU<Dtype>& lhs, const TensorGPU<Dtype>& rhs) {
    
    TensorGPU<Dtype> result(lhs.count());

    hypertea_gpu_add<Dtype>(lhs.count(), lhs.immutable_data(), rhs.immutable_data(), result.mutable_data());

  return result;
}
template TensorGPU<float> operator+(const TensorGPU<float>& a, const TensorGPU<float>& rhs);


template <typename Dtype>
TensorGPU<Dtype> operator+(const TensorGPU<Dtype>& lhs, const float rhs) {
    
    TensorGPU<Dtype> result(lhs.count());
    hypertea_cl_copy<Dtype>(lhs.count(), lhs.immutable_data(), result.mutable_data());
    hypertea_gpu_add_scalar<Dtype>(lhs.count(), rhs, result.mutable_data());
  return result;
}
template TensorGPU<float> operator+(const TensorGPU<float>& a, const float rhs);




template <typename Dtype>
TensorGPU<Dtype> operator-(const TensorGPU<Dtype>& lhs, const TensorGPU<Dtype>& rhs) {
    
    TensorGPU<Dtype> result(lhs.count());

    hypertea_gpu_sub<Dtype>(lhs.count(), lhs.immutable_data(), rhs.immutable_data(), result.mutable_data());

  return result;
}
template TensorGPU<float> operator-(const TensorGPU<float>& a, const TensorGPU<float>& rhs);


template <typename Dtype>
TensorGPU<Dtype> operator-(const TensorGPU<Dtype>& lhs, const float rhs) {
    
    TensorGPU<Dtype> result(lhs.count());
    hypertea_cl_copy<Dtype>(lhs.count(), lhs.immutable_data(), result.mutable_data());
    hypertea_gpu_add_scalar<Dtype>(lhs.count(), -rhs, result.mutable_data());
  return result;
}
template TensorGPU<float> operator-(const TensorGPU<float>& a, const float rhs);




template <typename Dtype>
TensorGPU<Dtype> operator*(const TensorGPU<Dtype>& lhs, float rhs) {
    
    TensorGPU<Dtype> result(lhs.count());
    hypertea_gpu_scale<Dtype>(lhs.count(), rhs, lhs.immutable_data(), result.mutable_data());
  return result;
}
template TensorGPU<float> operator*(const TensorGPU<float>& a, const float rhs);
} //namespace hypertea