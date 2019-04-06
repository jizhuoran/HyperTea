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

	explicit TensorGPU(int count) {
		data_.reset((void*)clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, count * sizeof(Dtype), NULL, NULL), [=](void *ptr){clReleaseMemObject((cl_mem) ptr);});
		this->count_ = count;
	}

	explicit TensorGPU(int count, Dtype value);
	explicit TensorGPU(std::vector<Dtype> data);
	explicit TensorGPU(cl_mem data_ptr, int count, bool shared = false);

	TensorGPU& copy_data(const TensorGPU & other);
 	TensorGPU duplicate();

	virtual ~TensorGPU() {}
	
	cl_mem mutable_data() const { return (cl_mem)data_.get(); }
	const cl_mem immutable_data() const { return (cl_mem)data_.get(); }

	TensorGPU<Dtype> sub_view(unsigned int offset, unsigned int size, cl_mem_flags flags = CL_MEM_READ_WRITE);
	std::vector<TensorGPU<Dtype> > chunked_tensors(int chunck_num, cl_mem_flags flags = CL_MEM_READ_WRITE);

	std::shared_ptr<Dtype> debug_gtest_cpu_data() const;

	TensorGPU& operator+=(const TensorGPU & other) {return inplace_gpu_add(other, *this); }
	TensorGPU& operator+=(const float other) {return inplace_gpu_add_scalar(*this, other); }
	TensorGPU& operator-=(const TensorGPU & other) {return inplace_gpu_sub(other, *this); }
	TensorGPU& operator-=(const float other) {return inplace_gpu_sub_scalar(*this, other); }
	TensorGPU& operator*=(const TensorGPU & other) {return inplace_gpu_mul(other, *this); }
	TensorGPU& operator*=(const float other) {return inplace_gpu_mul_scalar(*this, other); }
	TensorGPU& operator/=(const TensorGPU & other) {return inplace_gpu_div(other, *this); }
	TensorGPU& operator/=(const float other) {return inplace_gpu_div_scalar(*this, other); }

	TensorGPU& sigmoid() {return inplace_gpu_sigmoid(*this); }
	TensorGPU& tanh() {return inplace_gpu_tanh(*this); }
	TensorGPU& abs() {return inplace_gpu_abs(*this); }
	TensorGPU& exp() {return inplace_gpu_exp(*this); }
	TensorGPU& log() {return inplace_gpu_log(*this); }
	TensorGPU& sqr() {return inplace_gpu_sqr(*this); }
	TensorGPU& sqrt() {return inplace_gpu_sqrt(*this); }

	TensorGPU& set(const Dtype e) {return inplace_gpu_set(*this, e); }
	TensorGPU& powx(const float e) {return inplace_gpu_powx(*this, e); }
	TensorGPU& elu(const float e) {return inplace_gpu_elu(*this, e); }
	TensorGPU& relu(const float e) {return inplace_gpu_relu(*this, e); }

private:

	TensorGPU();

	std::shared_ptr<void> data_;

};



template <typename Dtype>
class TensorCPU : public Tensor<Dtype>
{
public:

	explicit TensorCPU(int count) {
		data_.reset(new Dtype[count], std::default_delete<Dtype[]>() );
		this->count_ = count;
	}

	explicit TensorCPU(int count, Dtype value);
	explicit TensorCPU(std::vector<Dtype> data);
	explicit TensorCPU(Dtype* data_ptr, int count, bool shared = false);



	TensorCPU& copy_data(const TensorCPU & other);
	TensorCPU duplicate() const;


	virtual ~TensorCPU() {}


	TensorCPU<Dtype> sub_view(unsigned int offset, unsigned int size);
	std::vector<TensorCPU<Dtype> > chunked_tensors(int chunck_num);


	Dtype* mutable_data() const {return data_.get();}
	const Dtype* immutable_data() const {return data_.get();}

	std::shared_ptr<Dtype> duplicate_data() const;

	std::shared_ptr<Dtype> debug_gtest_cpu_data() const {
		return duplicate_data();
	}


	// TensorCPU add(TensorCPU & other, Dtype alpha=1);

	TensorCPU& operator+=(const TensorCPU & other) {return inplace_cpu_add(other, *this); }
	TensorCPU& operator+=(const float other) {return inplace_cpu_add_scalar(*this, other); }
	TensorCPU& operator-=(const TensorCPU & other) {return inplace_cpu_sub(other, *this); }
	TensorCPU& operator-=(const float other) {return inplace_cpu_sub_scalar(*this, other); }
	TensorCPU& operator*=(const TensorCPU & other) {return inplace_cpu_mul(other, *this); }
	TensorCPU& operator*=(const float other) {return inplace_cpu_mul_scalar(*this, other); }
	TensorCPU& operator/=(const TensorCPU & other) {return inplace_cpu_div(other, *this); }
	TensorCPU& operator/=(const float other) {return inplace_cpu_div_scalar(*this, other); }


	TensorCPU& sigmoid() {return inplace_cpu_sigmoid(*this); }
	TensorCPU& tanh() {return inplace_cpu_tanh(*this); }
	TensorCPU& abs() {return inplace_cpu_abs(*this); }
	TensorCPU& exp() {return inplace_cpu_exp(*this); }
	TensorCPU& log() {return inplace_cpu_log(*this); }
	TensorCPU& sqr() {return inplace_cpu_sqr(*this); }
	TensorCPU& sqrt() {return inplace_cpu_sqrt(*this); }

	TensorCPU& set(const Dtype e) {return inplace_cpu_set(*this, e); }
	TensorCPU& powx(const float e) {return inplace_cpu_powx(*this, e); }
	TensorCPU& elu(const float e) {return inplace_cpu_elu(*this, e); }
	TensorCPU& relu(const float e) {return inplace_cpu_relu(*this, e); }


private:

	TensorCPU();
	std::shared_ptr<Dtype> data_;


};




}

#endif //HYPERTEA_TENSOR_H_