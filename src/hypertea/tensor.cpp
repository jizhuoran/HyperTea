#include "hypertea/tensor.hpp"
#include "hypertea/util/math_functions.hpp"


namespace hypertea {


template <typename Dtype>
TensorCPU<Dtype>::TensorCPU(int count, Dtype value) {
    data_.reset(new Dtype[count], std::default_delete<Dtype[]>() );

    hypertea_set(count, value, data_.get());

    this->count_ = count;
}
template TensorCPU<float>::TensorCPU(int count, float value);



template <typename Dtype>
TensorCPU<Dtype>& TensorCPU<Dtype>::copy_data(const TensorCPU & other) {
  hypertea_copy<Dtype>(this->count(), other.immutable_data(), this->mutable_data());
  return *this;
}
template TensorCPU<float>& TensorCPU<float>::copy_data(const TensorCPU<float> & other);


template <typename Dtype>
TensorCPU<Dtype> TensorCPU<Dtype>::duplicate() const{
  TensorCPU temp = TensorCPU(this->count_);
  temp.copy_data(*this);
  return temp;
}
template TensorCPU<float> TensorCPU<float>::duplicate() const;



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



// template <typename Dtype>
// TensorCPU<Dtype>& TensorCPU<Dtype>::operator+=(const TensorCPU<Dtype> & other) {
// 	hypertea_add<Dtype>(this->count(), this->immutable_data(), other.immutable_data(), this->mutable_data());
//   	return *this;
// }
// template TensorCPU<float>& TensorCPU<float>::operator+=(const TensorCPU<float> & other);


// template <typename Dtype>
// TensorCPU<Dtype>& TensorCPU<Dtype>::operator+=(const Dtype other) {
//   hypertea_add_scalar<Dtype>(this->count(), other, this->mutable_data());
//     return *this;
// }
// template TensorCPU<float>& TensorCPU<float>::operator+=(const float other);




// template <typename Dtype>
// TensorCPU<Dtype>& TensorCPU<Dtype>::operator*=(const Dtype other) {
//   hypertea_scal<Dtype>(this->count(), other, this->mutable_data());
//     return *this;
// }
// template TensorCPU<float>& TensorCPU<float>::operator*=(const float other);
 
 




// template <typename Dtype>
// TensorCPU<Dtype> operator+(const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs) {
    
//     assert(lhs.count() == rhs.count());

//     TensorCPU<Dtype> result(lhs.count());
//     hypertea_add<Dtype>(lhs.count(), lhs.immutable_data(), rhs.immutable_data(), result.mutable_data());

// 	return result;
// }
// template TensorCPU<float> operator+(const TensorCPU<float>& lhs, const TensorCPU<float>& rhs);


// template <typename Dtype>
// TensorCPU<Dtype> operator+(const TensorCPU<Dtype>& lhs, const float rhs) {
    
//     TensorCPU<Dtype> result(lhs.count());
//     hypertea_copy(lhs.count(), lhs.immutable_data(), result.mutable_data());
//     hypertea_add_scalar<Dtype>(lhs.count(), rhs, result.mutable_data());
//   return result;
// }
// template TensorCPU<float> operator+(const TensorCPU<float>& lhs, const float rhs);




// template <typename Dtype>
// TensorCPU<Dtype> operator-(const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs) {
    
//     assert(lhs.count() == rhs.count());

//     TensorCPU<Dtype> result(lhs.count());
//     hypertea_sub<Dtype>(lhs.count(), lhs.immutable_data(), rhs.immutable_data(), result.mutable_data());

// 	return result;
// }
// template TensorCPU<float> operator-(const TensorCPU<float>& lhs, const TensorCPU<float>& rhs);


// template <typename Dtype>
// TensorCPU<Dtype> operator-(const TensorCPU<Dtype>& lhs, const float rhs) {
    
//     TensorCPU<Dtype> result(lhs.count());
//     hypertea_copy(lhs.count(), lhs.immutable_data(), result.mutable_data());
//     hypertea_add_scalar<Dtype>(lhs.count(), -rhs, result.mutable_data());
//   return result;
// }
// template TensorCPU<float> operator-(const TensorCPU<float>& lhs, const float rhs);




// template <typename Dtype>
// TensorCPU<Dtype> operator*(const TensorCPU<Dtype>& lhs, float rhs) {
    
//     TensorCPU<Dtype> result(lhs.count());
//     hypertea_cpu_scale<Dtype>(lhs.count(), rhs, lhs.immutable_data(), result.mutable_data());
//   return result;
// }
// template TensorCPU<float> operator*(const TensorCPU<float>& a, const float rhs);


// template <typename Dtype>
// TensorCPU<Dtype> operator*(const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs) {
    
//   assert(lhs.count() == rhs.count());

//   TensorCPU<Dtype> result(lhs.count());
//   hypertea_mul(lhs.count(), lhs.immutable_data(), rhs.immutable_data(), result.mutable_data());

//   return result;
// }
// template TensorCPU<float> operator*(const TensorCPU<float>& lhs, const TensorCPU<float>& rhs);















template <typename Dtype>
TensorGPU<Dtype>::TensorGPU(int count, Dtype value) {

  data_.reset(
    (void*)clCreateBuffer(OpenCLHandler::Get().context, 
      CL_MEM_READ_WRITE,
      count * sizeof(Dtype), 
      NULL, NULL
    ), 
    [=](void *ptr){clReleaseMemObject((cl_mem) ptr);}
  );
  this->set(value);
  this->count_ = count;
}
template TensorGPU<float>::TensorGPU(int count, float value);
template TensorGPU<half>::TensorGPU(int count, half value);



template <typename Dtype>
TensorGPU<Dtype>::TensorGPU(cl_mem data_ptr, int count, bool shared) {

    if (shared) {
      data_.reset((void*)data_ptr, [](void *ptr){});
    } else {
      data_.reset((void*)data_ptr, [=](void *ptr){clReleaseMemObject((cl_mem) ptr);});
    }
    this->count_ = count;
}
template TensorGPU<float>::TensorGPU(cl_mem data_ptr, int count, bool shared);
template TensorGPU<half>::TensorGPU(cl_mem data_ptr, int count, bool shared);



template <typename Dtype>
TensorGPU<Dtype>::TensorGPU(std::vector<Dtype> data) {
  
  data_.reset(
    (void*)clCreateBuffer(
      OpenCLHandler::Get().context, 
      CL_MEM_COPY_HOST_PTR|CL_MEM_READ_WRITE,  
      data.size() * sizeof(Dtype),
      data.data(),
      NULL
    ), 
    [=](void *ptr){clReleaseMemObject((cl_mem) ptr);}
  );
  this->count_ = data.size();
}
template TensorGPU<float>::TensorGPU(std::vector<float> data);
template TensorGPU<half>::TensorGPU(std::vector<half> data);


template <typename Dtype>
TensorGPU<Dtype>& TensorGPU<Dtype>::copy_data(const TensorGPU & other) {
  hypertea_cl_copy<Dtype>(this->count(), other.immutable_data(), this->mutable_data());
  return *this;
}
template TensorGPU<float>& TensorGPU<float>::copy_data(const TensorGPU<float> & other);
template TensorGPU<half>& TensorGPU<half>::copy_data(const TensorGPU<half> & other);


template <typename Dtype>
TensorGPU<Dtype> TensorGPU<Dtype>::duplicate() {
  TensorGPU temp = TensorGPU(this->count_);
  temp.copy_data(*this);
  return temp;
}
template TensorGPU<float> TensorGPU<float>::duplicate();
template TensorGPU<half> TensorGPU<half>::duplicate();






template <typename Dtype>
TensorGPU<Dtype> TensorGPU<Dtype>::sub_view(unsigned int offset, unsigned int size, cl_mem_flags flags) {
  cl_int ret;
  cl_buffer_region region{offset * sizeof(Dtype), size * sizeof(Dtype)};
  auto temp = clCreateSubBuffer((cl_mem)data_.get(), flags, CL_BUFFER_CREATE_TYPE_REGION, &region, &ret); 
  OPENCL_CHECK(ret);
  return TensorGPU<Dtype>(temp, size, true);
}
template TensorGPU<float> TensorGPU<float>::sub_view(unsigned int offset, unsigned int size, cl_mem_flags flags);
template TensorGPU<half> TensorGPU<half>::sub_view(unsigned int offset, unsigned int size, cl_mem_flags flags);




template <typename Dtype>
std::vector<TensorGPU<Dtype> > TensorGPU<Dtype>::chunked_tensors(int chunck_num, cl_mem_flags flags) {
    
  size_t chunck_count = this->count_ / chunck_num;
  size_t chunck_size = chunck_count * sizeof(Dtype);


  cl_int ret;
  cl_buffer_region region{0, chunck_size};

  std::vector<TensorGPU<Dtype> > tensors;
  for (int i = 0; i < chunck_num; ++i) {
    tensors.push_back(
      TensorGPU<Dtype>(
        clCreateSubBuffer((cl_mem)data_.get(), flags, CL_BUFFER_CREATE_TYPE_REGION, &region, &ret),
        chunck_count,
        true
      )
    );
        OPENCL_CHECK(ret);
        region.origin += chunck_size;

  }

      return tensors;
}

template std::vector<TensorGPU<float> > TensorGPU<float>::chunked_tensors(int chunck_num, cl_mem_flags flags);
template std::vector<TensorGPU<half> > TensorGPU<half>::chunked_tensors(int chunck_num, cl_mem_flags flags);





template <typename Dtype>
std::shared_ptr<Dtype> TensorGPU<Dtype>::debug_gtest_cpu_data() const {
  auto cpu_data = std::shared_ptr<Dtype>(new Dtype[this->count_], std::default_delete<Dtype[]>());
  OPENCL_CHECK(clEnqueueReadBuffer(OpenCLHandler::Get().commandQueue, (cl_mem)data_.get(), CL_TRUE, 0, sizeof(Dtype) * this->count_, cpu_data.get(), 0, NULL, NULL));
  return cpu_data;
}

template std::shared_ptr<float> TensorGPU<float>::debug_gtest_cpu_data() const;
template std::shared_ptr<half> TensorGPU<half>::debug_gtest_cpu_data() const;



// template <typename Dtype>
// TensorGPU<Dtype>& TensorGPU<Dtype>::sigmoid() {
//   hypertea_gpu_sigmoid<Dtype>(this->count(), this->mutable_data(), this->mutable_data());
//   return *this;
// }
// template TensorGPU<float>& TensorGPU<float>::sigmoid();
// template TensorGPU<half>& TensorGPU<half>::sigmoid();



// template <typename Dtype>
// TensorGPU<Dtype>& TensorGPU<Dtype>::tanh() {
//   hypertea_gpu_tanh<Dtype>(this->count(), this->mutable_data(), this->mutable_data());
//   return *this;
// }
// template TensorGPU<float>& TensorGPU<float>::tanh();
// template TensorGPU<half>& TensorGPU<half>::tanh();


// template <typename Dtype>
// TensorGPU<Dtype>& TensorGPU<Dtype>::abs() {
//   hypertea_gpu_abs<Dtype>(this->count(), this->mutable_data(), this->mutable_data());
//   return *this;
// }
// template TensorGPU<float>& TensorGPU<float>::abs();
// template TensorGPU<half>& TensorGPU<half>::abs();


// template <typename Dtype>
// TensorGPU<Dtype>& TensorGPU<Dtype>::exp() {
//   hypertea_gpu_exp<Dtype>(this->count(), this->mutable_data(), this->mutable_data());
//   return *this;
// }
// template TensorGPU<float>& TensorGPU<float>::exp();
// template TensorGPU<half>& TensorGPU<half>::exp();


// template <typename Dtype>
// TensorGPU<Dtype>& TensorGPU<Dtype>::log() {
//   hypertea_gpu_log<Dtype>(this->count(), this->mutable_data(), this->mutable_data());
//   return *this;
// }
// template TensorGPU<float>& TensorGPU<float>::log();
// template TensorGPU<half>& TensorGPU<half>::log();


// template <typename Dtype>
// TensorGPU<Dtype>& TensorGPU<Dtype>::powx(float e) {
//   hypertea_gpu_powx<Dtype>(this->count(), this->mutable_data(), e, this->mutable_data());
//   return *this;
// }
// template TensorGPU<float>& TensorGPU<float>::powx(float e);
// template TensorGPU<half>& TensorGPU<half>::powx(float e);


// template <typename Dtype>
// TensorGPU<Dtype>& TensorGPU<Dtype>::sqrt() {
//   hypertea_gpu_sqrt<Dtype>(this->count(), this->mutable_data(), this->mutable_data());
//   return *this;
// }
// template TensorGPU<float>& TensorGPU<float>::sqrt();
// template TensorGPU<half>& TensorGPU<half>::sqrt();




// template <typename Dtype>
// TensorGPU<Dtype>& TensorGPU<Dtype>::operator+=(const TensorGPU<Dtype> & other) {
//   return inplace_gpu_add(other, *this);
// }
// template TensorGPU<float>& TensorGPU<float>::operator+=(const TensorGPU<float> & other);
// template TensorGPU<half>& TensorGPU<half>::operator+=(const TensorGPU<half> & other);


// template <typename Dtype>
// TensorGPU<Dtype>& TensorGPU<Dtype>::operator+=(const float other) {
//   hypertea_gpu_add_scalar<Dtype>(this->count(), other, this->mutable_data());
//     return *this;
// }
// template TensorGPU<float>& TensorGPU<float>::operator+=(const float other);
// template TensorGPU<half>& TensorGPU<half>::operator+=(const float other);

// // template <typename Dtype>
// // TensorGPU<Dtype>& TensorGPU<Dtype>::operator-=(const TensorGPU<Dtype> & other) {
// //   hypertea_gpu_sub<Dtype>(this->count(), this->immutable_data(), other.immutable_data(), this->mutable_data());
// //     return *this;
// // }
// // template TensorGPU<float>& TensorGPU<float>::operator-=(const TensorGPU<float> & other);
// // template TensorGPU<half>& TensorGPU<half>::operator-=(const TensorGPU<half> & other);


// template <typename Dtype>
// TensorGPU<Dtype>& TensorGPU<Dtype>::operator-=(const float other) {
//   hypertea_gpu_add_scalar<Dtype>(this->count(), -other, this->mutable_data());
//     return *this;
// }
// template TensorGPU<float>& TensorGPU<float>::operator-=(const float other);
// template TensorGPU<half>& TensorGPU<half>::operator-=(const float other);


// // template <typename Dtype>
// // TensorGPU<Dtype>& TensorGPU<Dtype>::operator*=(const TensorGPU<Dtype> & other) {
// //   hypertea_gpu_mul<Dtype>(this->count(), this->immutable_data(), other.immutable_data(), this->mutable_data());
// //   return *this;
// // }
// // template TensorGPU<float>& TensorGPU<float>::operator*=(const TensorGPU<float> & other);
// // template TensorGPU<half>& TensorGPU<half>::operator*=(const TensorGPU<half> & other);


// template <typename Dtype>
// TensorGPU<Dtype>& TensorGPU<Dtype>::operator*=(const float other) {
//   hypertea_gpu_scal<Dtype>(this->count(), other, this->mutable_data());
//   return *this;
// }
// template TensorGPU<float>& TensorGPU<float>::operator*=(const float other);
// template TensorGPU<half>& TensorGPU<half>::operator*=(const float other);


// template <typename Dtype>
// TensorGPU<Dtype>& TensorGPU<Dtype>::operator/=(const TensorGPU<Dtype> & other) {
//   hypertea_gpu_div<Dtype>(this->count(), this->immutable_data(), other.immutable_data(), this->mutable_data());
//   return *this;
// }
// template TensorGPU<float>& TensorGPU<float>::operator/=(const TensorGPU<float> & other);
// template TensorGPU<half>& TensorGPU<half>::operator/=(const TensorGPU<half> & other);


	
// template <typename Dtype>
// TensorGPU<Dtype> operator+(const TensorGPU<Dtype>& lhs, const TensorGPU<Dtype>& rhs) {
//   return gpu_add(lhs ,rhs);
// }
// template TensorGPU<float> operator+(const TensorGPU<float>& a, const TensorGPU<float>& rhs);
// template TensorGPU<half> operator+(const TensorGPU<half>& a, const TensorGPU<half>& rhs);


// template <typename Dtype>
// TensorGPU<Dtype> operator+(const TensorGPU<Dtype>& lhs, const float rhs) {
    
//     TensorGPU<Dtype> result(lhs.count());



//     hypertea_cl_copy<Dtype>(lhs.count(), lhs.immutable_data(), result.mutable_data());
//     hypertea_gpu_add_scalar<Dtype>(lhs.count(), rhs, result.mutable_data());
//   return result;
// }
// template TensorGPU<float> operator+(const TensorGPU<float>& a, const float rhs);
// template TensorGPU<half> operator+(const TensorGPU<half>& a, const float rhs);




// template <typename Dtype>
// TensorGPU<Dtype> operator-(const TensorGPU<Dtype>& lhs, const TensorGPU<Dtype>& rhs) {
    
//     TensorGPU<Dtype> result(lhs.count());

//     hypertea_gpu_sub<Dtype>(lhs.count(), lhs.immutable_data(), rhs.immutable_data(), result.mutable_data());

//   return result;
// }
// template TensorGPU<float> operator-(const TensorGPU<float>& a, const TensorGPU<float>& rhs);
// template TensorGPU<half> operator-(const TensorGPU<half>& a, const TensorGPU<half>& rhs);


// template <typename Dtype>
// TensorGPU<Dtype> operator-(const TensorGPU<Dtype>& lhs, const float rhs) {
    
//     TensorGPU<Dtype> result(lhs.count());
//     hypertea_cl_copy<Dtype>(lhs.count(), lhs.immutable_data(), result.mutable_data());
//     hypertea_gpu_add_scalar<Dtype>(lhs.count(), -rhs, result.mutable_data());
//   return result;
// }
// template TensorGPU<float> operator-(const TensorGPU<float>& a, const float rhs);
// template TensorGPU<half> operator-(const TensorGPU<half>& a, const float rhs);



// template <typename Dtype>
// TensorGPU<Dtype> operator*(const TensorGPU<Dtype>& lhs, const TensorGPU<Dtype>& rhs) {
    
//     TensorGPU<Dtype> result(lhs.count());

//     hypertea_gpu_mul<Dtype>(lhs.count(), lhs.immutable_data(), rhs.immutable_data(), result.mutable_data());

//   return result;
// }
// template TensorGPU<float> operator*(const TensorGPU<float>& a, const TensorGPU<float>& rhs);
// template TensorGPU<half> operator*(const TensorGPU<half>& a, const TensorGPU<half>& rhs);



// template <typename Dtype>
// TensorGPU<Dtype> operator*(const TensorGPU<Dtype>& lhs, float rhs) {
    
//     TensorGPU<Dtype> result(lhs.count());
//     hypertea_gpu_scale<Dtype>(lhs.count(), rhs, lhs.immutable_data(), result.mutable_data());
//   return result;
// }
// template TensorGPU<float> operator*(const TensorGPU<float>& a, const float rhs);
// template TensorGPU<half> operator*(const TensorGPU<half>& a, const float rhs);


// template <typename Dtype>
// TensorGPU<Dtype> operator/(const TensorGPU<Dtype>& lhs, const TensorGPU<Dtype>& rhs) {
    
//     TensorGPU<Dtype> result(lhs.count());

//     hypertea_gpu_div<Dtype>(lhs.count(), lhs.immutable_data(), rhs.immutable_data(), result.mutable_data());

//   return result;
// }
// template TensorGPU<float> operator/(const TensorGPU<float>& a, const TensorGPU<float>& rhs);
// template TensorGPU<half> operator/(const TensorGPU<half>& a, const TensorGPU<half>& rhs);
} //namespace hypertea