#ifndef HYPERTEA_TENSOR_H_
#define HYPERTEA_TENSOR_H_

#include <vector>
#include <assert.h>
#include <numeric>
#include "hypertea/common.hpp"

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



template<typename Dtype> TensorGPU<Dtype> operator+ (const TensorGPU<Dtype>& lhs, const TensorGPU<Dtype>& rhs);
template<typename Dtype> TensorGPU<Dtype> operator+ (const TensorGPU<Dtype>& lhs, const float rhs);
template<typename Dtype> TensorGPU<Dtype> operator- (const TensorGPU<Dtype>& lhs, const TensorGPU<Dtype>& rhs);
template<typename Dtype> TensorGPU<Dtype> operator- (const TensorGPU<Dtype>& lhs, const float rhs);

template<typename Dtype> TensorGPU<Dtype> operator* (const TensorGPU<Dtype>& lhs, const float rhs);

template <typename Dtype>
class Tensor
{

public:
	Tensor() {}
	~Tensor() {}
	
	const int size() const {return count_; }
	const int count() const {return count_; }
	const std::vector<int> shape() const {return ((shape_.size() == 0) ? std::vector<int>{count_}:shape_);}

	void reshape(std::vector<int> shape) {
		shape_ = shape;
	}


	// virtual Tensor add(Tensor & other, Dtype alpha=1);

	// virtual Tensor& operator+=(const Tensor & other);
	// virtual Tensor& operator+=(const Dtype other);


	// // Tensor& operator*=(const Tensor & other);
	// virtual Tensor& operator*=(const Dtype other);


protected:

	int count_ = 0;
	std::vector<int> shape_;
	// std::vector<int> shape_;

};


template <typename Dtype>
class TensorGPU : public Tensor<Dtype>
{
public:

	TensorGPU(int count) {
		data_.reset((void*)clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, count * sizeof(Dtype), NULL, NULL), [=](void *ptr){clReleaseMemObject((cl_mem) ptr);});
		this->count_ = count;
	}

	TensorGPU(std::vector<Dtype> data) {
		data_.reset((void*)clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_COPY_HOST_PTR|CL_MEM_READ_WRITE,  data.size() * sizeof(Dtype), data.data(), NULL), [=](void *ptr){clReleaseMemObject((cl_mem) ptr);});
		this->count_ = data.size();
	}

	TensorGPU(cl_mem data_ptr, int count) {
		data_.reset((void*)data_ptr, [=](void *ptr){clReleaseMemObject((cl_mem) ptr);});
		this->count_ = count;
	}

	~TensorGPU() {}
	

	cl_mem mutable_data() const {
		return (cl_mem)data_.get();
	}

	const cl_mem immutable_data() const {
		return (cl_mem)data_.get();
	}


	Dtype* debug_cpu_data() const {
		float* cpu_data = new float[this->count_];
		OPENCL_CHECK(clEnqueueReadBuffer(OpenCLHandler::Get().commandQueue, (cl_mem)data_.get(), CL_TRUE, 0, sizeof(Dtype) * this->count_, cpu_data, 0, NULL, NULL));
		return cpu_data;
	}


	TensorGPU& operator+=(const TensorGPU & other);
	TensorGPU& operator+=(const Dtype other);

	TensorGPU& operator*=(const Dtype other);


private:

	std::shared_ptr<void> data_;

	TensorGPU();

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
		this->shape_ = std::vector<int> {count};
	}

	TensorCPU(std::vector<Dtype> data, const std::vector<int>& shape = {}) {
		data_.reset(new Dtype[data.size()], std::default_delete<Dtype[]>() );
		memcpy(data_.get(), data.data(), data.size() * sizeof(Dtype));
		this->count_ = data.size();

		if(shape.size() == 0) {
			this->shape_ = std::vector<int> {data.size()};
		} else {
			this->shape_ = shape;
		}

	}
	
	TensorCPU(std::vector<int> shape) {
		this->shape_ = shape;
		this->count_ = std::accumulate(shape.begin(), shape.end(), 1);
		data_.reset(new Dtype[this->count_], std::default_delete<Dtype[]>() );
	}


	

	TensorCPU(Dtype* data_ptr, int count, const std::vector<int>& shape = {}) {
		data_.reset(data_ptr, std::default_delete<Dtype[]>() );
		this->count_ = count;

		if(shape.size() == 0) {
			this->shape_ = std::vector<int> {count};
		} else {
			this->shape_ = shape;
		}

	}
	TensorCPU(std::shared_ptr<Dtype> data_ptr, int count, const std::vector<int>& shape = {}) {
		data_ = data_ptr;
		this->count_ = count;

		if(shape.size() == 0) {
			this->shape_ = std::vector<int> {count};
		} else {
			this->shape_ = shape;
		}
	}

	// TensorCPU(const TensorCPU& other) {
	// 	data_ = other.duplicate_data();
	// 	this->count_ = other.count();
	// }

	~TensorCPU() {}


	Dtype* mutable_data() const {return data_.get();}
	const Dtype* immutable_data() const {return data_.get();}

	std::shared_ptr<Dtype> duplicate_data() const;




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