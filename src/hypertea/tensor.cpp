#include "hypertea/tensor.hpp"
#include "hypertea/util/math_functions.hpp"


namespace hypertea {


template <typename Dtype>
std::shared_ptr<Dtype> TensorCPU<Dtype>::duplicate_data() const {
  Dtype* t = new Dtype[this->count_];
  hypertea_copy(this->count_, data_.get(), t);
  return std::shared_ptr<Dtype>(t, std::default_delete<Dtype[]>());
}
template std::shared_ptr<float> TensorCPU<float>::duplicate_data() const;




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
template TensorCPU<float> operator+(const TensorCPU<float>& lhs, const TensorCPU<float>& rhs);


template <typename Dtype>
TensorCPU<Dtype> operator+(const TensorCPU<Dtype>& lhs, const float rhs) {
    
    TensorCPU<Dtype> result(lhs.count());
    hypertea_copy(lhs.count(), lhs.immutable_data(), result.mutable_data());
    hypertea_add_scalar<Dtype>(lhs.count(), rhs, result.mutable_data());
  return result;
}
template TensorCPU<float> operator+(const TensorCPU<float>& lhs, const float rhs);




template <typename Dtype>
TensorCPU<Dtype> operator-(const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs) {
    
    TensorCPU<Dtype> result(lhs.count());

    hypertea_sub<Dtype>(lhs.count(), lhs.immutable_data(), rhs.immutable_data(), result.mutable_data());

	return result;
}
template TensorCPU<float> operator-(const TensorCPU<float>& lhs, const TensorCPU<float>& rhs);


template <typename Dtype>
TensorCPU<Dtype> operator-(const TensorCPU<Dtype>& lhs, const float rhs) {
    
    TensorCPU<Dtype> result(lhs.count());
    hypertea_copy(lhs.count(), lhs.immutable_data(), result.mutable_data());
    hypertea_add_scalar<Dtype>(lhs.count(), -rhs, result.mutable_data());
  return result;
}
template TensorCPU<float> operator-(const TensorCPU<float>& lhs, const float rhs);




template <typename Dtype>
TensorCPU<Dtype> operator*(const TensorCPU<Dtype>& lhs, float rhs) {
    
    TensorCPU<Dtype> result(lhs.count());
    hypertea_cpu_scale<Dtype>(lhs.count(), rhs, lhs.immutable_data(), result.mutable_data());
  return result;
}
template TensorCPU<float> operator*(const TensorCPU<float>& a, const float rhs);


template <typename Dtype>
TensorCPU<Dtype> operator*(const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs) {
    
    assert(lhs.shape()[1] == rhs.shape()[0]);

    int M = lhs.shape()[0];
    int K = lhs.shape()[1];
    int N = rhs.shape()[1];

    TensorCPU<Dtype> result(M*N);
    result.reshape(std::vector<int> {M, N});

    hypertea_cpu_gemm<Dtype>(
      CblasNoTrans, CblasNoTrans,
      M, N, K,
      1,
      lhs.immutable_data(), 
      rhs.immutable_data(),
      0,
      result.mutable_data()
    );



    // hypertea_cpu_scale<Dtype>(lhs.count(), rhs, lhs.immutable_data(), result.mutable_data());
  return result;
}
template TensorCPU<float> operator*(const TensorCPU<float>& lhs, const TensorCPU<float>& rhs);



















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