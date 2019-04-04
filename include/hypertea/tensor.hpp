#ifndef HYPERTEA_TENSOR_H_
#define HYPERTEA_TENSOR_H_

#include <vector>
#include <assert.h>
#include <numeric>
#include "hypertea/common.hpp"
#include "hypertea/util/tensor_math_functions.hpp"

namespace hypertea {

template<typename Dtype> class Tensor;
template<typename Dtype> class TensorCPU;
template<typename Dtype> class TensorGPU;
template<typename Dtype> TensorCPU<Dtype> operator+ (const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs);
template<typename Dtype> TensorCPU<Dtype> operator+ (const TensorCPU<Dtype>& lhs, const float rhs);
template<typename Dtype> TensorCPU<Dtype> operator- (const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs);
template<typename Dtype> TensorCPU<Dtype> operator- (const TensorCPU<Dtype>& lhs, const float rhs);
template<typename Dtype> TensorCPU<Dtype> operator* (const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs);
template<typename Dtype> TensorCPU<Dtype> operator* (const TensorCPU<Dtype>& lhs, const float rhs);





template <typename Dtype>
class Tensor
{

public:
	Tensor() {}
	~Tensor() {}
	
	const int size() const {return count_; }
	const int count() const {return count_; }



	// virtual Tensor add(Tensor & other, Dtype alpha=1);

	// virtual Tensor& operator+=(const Tensor & other);
	// virtual Tensor& operator+=(const Dtype other);


	// // Tensor& operator*=(const Tensor & other);
	// virtual Tensor& operator*=(const Dtype other);


protected:

	int count_ = 0;
	// std::vector<int> shape_;

};


template <typename Dtype>
class TensorGPU : public Tensor<Dtype>
{
public:

	TensorGPU() {}

	TensorGPU(int count) {
		data_.reset((void*)clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, count * sizeof(Dtype), NULL, NULL), [=](void *ptr){clReleaseMemObject((cl_mem) ptr);});
		this->count_ = count;
	}

	TensorGPU(std::vector<Dtype> data) {
		data_.reset((void*)clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_COPY_HOST_PTR|CL_MEM_READ_WRITE,  data.size() * sizeof(Dtype), data.data(), NULL), [=](void *ptr){clReleaseMemObject((cl_mem) ptr);});
		this->count_ = data.size();
	}

	TensorGPU(cl_mem data_ptr, int count, bool shared = false) {
		if (shared) {
			data_.reset((void*)data_ptr, [](void *ptr){});
		} else {
			data_.reset((void*)data_ptr, [=](void *ptr){clReleaseMemObject((cl_mem) ptr);});
		}
		
		this->count_ = count;
	}

	// TensorGPU& operator=(TensorGPU other) {
 //        std::cout << "copy assignment of A\n";
 //        this->data_ = other.data_;
 //        return *this;
 //    }

	~TensorGPU() {}
	

	cl_mem mutable_data() const {
		return (cl_mem)data_.get();
	}

	const cl_mem immutable_data() const {
		return (cl_mem)data_.get();
	}


	const size_t reference_count() const {

		cl_uint refer_count;

		OPENCL_CHECK(
			clGetMemObjectInfo (
				data_.get(),
			 	CL_MEM_REFERENCE_COUNT,
			 	sizeof(cl_uint),
			 	&refer_count,
			 	nullptr
 			)
 		);

 		return refer_count;
	}


	cl_mem sub_cl_view(unsigned int offset, unsigned int size, cl_mem_flags flags = CL_MEM_READ_WRITE) {
		cl_int ret;
		cl_buffer_region region{offset * sizeof(Dtype), size * sizeof(Dtype)};
        auto temp = clCreateSubBuffer((cl_mem)data_.get(), flags, CL_BUFFER_CREATE_TYPE_REGION, &region, &ret); 
        OPENCL_CHECK(ret);
        return temp;
	}


	TensorGPU<Dtype> sub_view(unsigned int offset, unsigned int size, cl_mem_flags flags = CL_MEM_READ_WRITE) {
		cl_int ret;
		cl_buffer_region region{offset * sizeof(Dtype), size * sizeof(Dtype)};
        auto temp = clCreateSubBuffer((cl_mem)data_.get(), flags, CL_BUFFER_CREATE_TYPE_REGION, &region, &ret); 
        OPENCL_CHECK(ret);
        return TensorGPU<Dtype>(temp, size, true);
	}

	std::vector<TensorGPU<Dtype> > chunked_tensors(int chunck_num, cl_mem_flags flags = CL_MEM_READ_WRITE) {
		
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


	const Dtype* debug_cpu_data() const {
		Dtype* cpu_data = new Dtype[this->count_];
		OPENCL_CHECK(clEnqueueReadBuffer(OpenCLHandler::Get().commandQueue, (cl_mem)data_.get(), CL_TRUE, 0, sizeof(Dtype) * this->count_, cpu_data, 0, NULL, NULL));
		return cpu_data;
	}


	std::shared_ptr<Dtype> cpu_data_gtest() const {

		auto cpu_data = std::shared_ptr<Dtype>(new Dtype[this->count_], std::default_delete<Dtype[]>());
		OPENCL_CHECK(clEnqueueReadBuffer(OpenCLHandler::Get().commandQueue, (cl_mem)data_.get(), CL_TRUE, 0, sizeof(Dtype) * this->count_, cpu_data.get(), 0, NULL, NULL));
		return cpu_data;
	}


	TensorGPU& operator+=(const TensorGPU & other) {return inplace_gpu_add(other, *this); }
	TensorGPU& operator+=(const float other) {return inplace_gpu_add_scalar(*this, other); }
	TensorGPU& operator-=(const TensorGPU & other) {return inplace_gpu_sub(other, *this); }
	TensorGPU& operator-=(const float other) {return inplace_gpu_sub_scalar(*this, other); }
	TensorGPU& operator*=(const TensorGPU & other) {return inplace_gpu_mul(other, *this); }
	TensorGPU& operator*=(const float other) {return inplace_gpu_mul_scalar(*this, other); }
	TensorGPU& operator/=(const TensorGPU & other) {return inplace_gpu_div(other, *this); }
	TensorGPU& operator/=(const float other) {return inplace_gpu_div_scalar(*this, other); }



	TensorGPU& copy_data(const TensorGPU & other);


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

	std::shared_ptr<void> data_;

	// TensorGPU();

	// virtual Tensor add(Tensor & other, Dtype alpha=1) {}

	// virtual Tensor& operator+=(const Tensor & other) {}
	// virtual Tensor& operator+=(const Dtype other) {}

	// virtual Tensor& operator*=(const Dtype other) {}


};



template <typename Dtype>
class TensorCPU : public Tensor<Dtype>
{
public:

	TensorCPU(int count) {
		data_.reset(new Dtype[count], std::default_delete<Dtype[]>() );
		this->count_ = count;
	}

	TensorCPU(int count, Dtype value);

	TensorCPU(std::vector<Dtype> data) {
		data_.reset(new Dtype[data.size()], std::default_delete<Dtype[]>() );
		memcpy(data_.get(), data.data(), data.size() * sizeof(Dtype));
		this->count_ = data.size();
	}
	
	TensorCPU(std::vector<int> shape) {
		this->count_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
		data_.reset(new Dtype[this->count_], std::default_delete<Dtype[]>() );
	}


	

	TensorCPU(Dtype* data_ptr, int count) {
		data_.reset(data_ptr, std::default_delete<Dtype[]>() );
		this->count_ = count;

	}
	TensorCPU(std::shared_ptr<Dtype> data_ptr, int count) {
		data_ = data_ptr;
		this->count_ = count;
	}

	~TensorCPU() {}


	Dtype* mutable_data() const {return data_.get();}
	const Dtype* immutable_data() const {return data_.get();}

	std::shared_ptr<Dtype> duplicate_data() const;

	// const Dtype* cpu_data_gtest() const {
	// 	return data_.get();
	// }


	std::shared_ptr<Dtype> cpu_data_gtest() const {
		return duplicate_data();
	}


	TensorCPU add(TensorCPU & other, Dtype alpha=1);

	TensorCPU& operator+=(const TensorCPU & other);
	TensorCPU& operator+=(const Dtype other);

	TensorCPU& operator*=(const Dtype other);


	friend TensorCPU<Dtype> operator+ <>(const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs);
	friend TensorCPU<Dtype> operator+ <>(const TensorCPU<Dtype>& lhs, const float rhs);
	friend TensorCPU<Dtype> operator- <>(const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs);
	friend TensorCPU<Dtype> operator- <>(const TensorCPU<Dtype>& lhs, const float rhs);
	friend TensorCPU<Dtype> operator* <>(const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs);
	friend TensorCPU<Dtype> operator* <>(const TensorCPU<Dtype>& lhs, const float rhs);

private:
	std::shared_ptr<Dtype> data_;

	TensorCPU();

};




}

#endif //HYPERTEA_TENSOR_H_