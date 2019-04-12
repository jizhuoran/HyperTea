#ifndef HYPERTEA_TENSOR_H_
#define HYPERTEA_TENSOR_H_

#include <vector>
#include <assert.h>
#include <numeric>
#include "hypertea/common.hpp"
#include "hypertea/util/tensor_gpu_math_func.hpp"
#include "hypertea/util/tensor_cpu_math_func.hpp"

namespace hypertea {


template <typename Dtype>
class Tensor
{

public:
	Tensor() {}
	virtual ~Tensor() {}
	
	const int size() const {return count_; }
	const int count() const {return count_; }

protected:

	int count_ = 0;

};


template <typename Dtype>
class TensorGPU : public Tensor<Dtype>
{
public:

	TensorGPU() = delete;

	explicit TensorGPU(int count) {
		data_.reset((void*)clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, count * sizeof(Dtype), NULL, NULL), [=](void *ptr){clReleaseMemObject((cl_mem) ptr);});
		this->count_ = count;
	}

	explicit TensorGPU(int count, Dtype value);
	explicit TensorGPU(std::vector<Dtype> data);
	explicit TensorGPU(cl_mem data_ptr, int count, bool shared = false);

	TensorGPU& copy_data(const TensorGPU & other);
 	TensorGPU duplicate() const;

 	void copy_to_ptr(void* ptr) const {
 		OPENCL_CHECK(
 			clEnqueueReadBuffer(
 				OpenCLHandler::Get().commandQueue, 
 				mutable_data(), CL_TRUE, 
 				0, this->count_ * sizeof(Dtype), 
 				ptr, 
 				0, nullptr, nullptr
 			)
 		);
 	}

	void copy_from_ptr(void* ptr) const {
 		OPENCL_CHECK(
 			clEnqueueWriteBuffer(
 				OpenCLHandler::Get().commandQueue, 
 				mutable_data(), CL_TRUE, 
 				0, this->count_ * sizeof(Dtype), 
 				ptr, 
 				0, nullptr, nullptr
 			)
 		);
 	}

	virtual ~TensorGPU() {}
	
	cl_mem mutable_data() const { return (cl_mem)data_.get(); }
	const cl_mem immutable_data() const { return (cl_mem)data_.get(); }

	TensorGPU<Dtype> sub_view(unsigned int offset, unsigned int size, cl_mem_flags flags = CL_MEM_READ_WRITE);
	std::vector<TensorGPU<Dtype> > chunked_tensors(int chunck_num, cl_mem_flags flags = CL_MEM_READ_WRITE);

	std::shared_ptr<Dtype> debug_gtest_cpu_data() const;

	TensorGPU& operator+=(const TensorGPU & other) {return inplace_add(other, *this); }
	TensorGPU& operator+=(const float other) {return inplace_add_scalar(*this, other); }
	TensorGPU& operator-=(const TensorGPU & other) {return inplace_sub(other, *this); }
	TensorGPU& operator-=(const float other) {return inplace_sub_scalar(*this, other); }
	TensorGPU& operator*=(const TensorGPU & other) {return inplace_mul(other, *this); }
	TensorGPU& operator*=(const float other) {return inplace_mul_scalar(*this, other); }
	TensorGPU& operator/=(const TensorGPU & other) {return inplace_div(other, *this); }
	TensorGPU& operator/=(const float other) {return inplace_div_scalar(*this, other); }

	TensorGPU& set(const Dtype e) {return inplace_set(*this, e); }

private:
	std::shared_ptr<void> data_;

};



template <typename Dtype>
class TensorCPU : public Tensor<Dtype>
{
public:

	TensorCPU() = delete;
	explicit TensorCPU(int count) {
		data_.reset(new Dtype[count], std::default_delete<Dtype[]>() );
		this->count_ = count;
	}

	explicit TensorCPU(int count, Dtype value);
	explicit TensorCPU(std::vector<Dtype> data);
	explicit TensorCPU(Dtype* data_ptr, int count, bool shared = false);



	TensorCPU& copy_data(const TensorCPU & other);
	TensorCPU duplicate() const;

	void copy_to_ptr(void* ptr) const {
 		memcpy(ptr, immutable_data(), this->count_ * sizeof(Dtype));
 	}

	void copy_from_ptr(void* ptr) const {
 		memcpy(mutable_data(), ptr, this->count_ * sizeof(Dtype));
 	}

	virtual ~TensorCPU() {}


	TensorCPU<Dtype> sub_view(unsigned int offset, unsigned int size);
	std::vector<TensorCPU<Dtype> > chunked_tensors(int chunck_num);


	Dtype* mutable_data() const {return data_.get();}
	const Dtype* immutable_data() const {return data_.get();}

	std::shared_ptr<Dtype> duplicate_data() const;

	std::shared_ptr<Dtype> debug_gtest_cpu_data() const {
		return duplicate_data();
	}



	TensorCPU& operator+=(const TensorCPU & other) {return inplace_add(other, *this); }
	TensorCPU& operator+=(const float other) {return inplace_add_scalar(*this, other); }
	TensorCPU& operator-=(const TensorCPU & other) {return inplace_sub(other, *this); }
	TensorCPU& operator-=(const float other) {return inplace_sub_scalar(*this, other); }
	TensorCPU& operator*=(const TensorCPU & other) {return inplace_mul(other, *this); }
	TensorCPU& operator*=(const float other) {return inplace_mul_scalar(*this, other); }
	TensorCPU& operator/=(const TensorCPU & other) {return inplace_div(other, *this); }
	TensorCPU& operator/=(const float other) {return inplace_div_scalar(*this, other); }

	TensorCPU& set(const Dtype e) {return inplace_set(*this, e); }

private:

	std::shared_ptr<Dtype> data_;


};




}

#endif //HYPERTEA_TENSOR_H_