#ifndef HYPERTEA_TENSOR_H_
#define HYPERTEA_TENSOR_H_

#include <vector>
#include "hypertea/util/math_functions.hpp"

namespace hypertea {

template<typename Dtype> class Tensor;
template<typename Dtype> Tensor<Dtype> operator+ (const Tensor<Dtype>& lhs, const Tensor<Dtype>& rhs);
template<typename Dtype> Tensor<Dtype> operator- (const Tensor<Dtype>& lhs, const Tensor<Dtype>& rhs);

template <typename Dtype>
class Tensor
{

public:
	Tensor() {}
	Tensor(int count) {
		data_.reset(new Dtype[count], std::default_delete<Dtype[]>() );
		count_ = count;
	}
	Tensor(std::vector<Dtype> data) {
		data_.reset(new Dtype[data.size()], std::default_delete<Dtype[]>() );
		memcpy(data_.get(), data.data(), data.size() * sizeof(Dtype));
		count_ = data.size();
	}
	Tensor(Dtype* data_ptr, int count) {
		data_.reset(data_ptr, std::default_delete<Dtype[]>() );
		count_ = count;
	}
	Tensor(std::shared_ptr<Dtype> data_ptr, int count) {
		data_ = data_ptr;
		count_ = count;
	}

	~Tensor() {}
	
	// std::vector<int> shape() {return shape_;}

	Dtype* data() const {return data_.get();}
	const Dtype* immutable_data() const {return data_.get();}



	const int size() const {return count_; }
	const int count() const {return count_; }




	Tensor add(Tensor & other, Dtype alpha=1);


	Tensor& operator+=(const Tensor & other);

	friend Tensor<Dtype> operator+ <>(const Tensor<Dtype>& lhs, const Tensor<Dtype>& rhs);
	friend Tensor<Dtype> operator- <>(const Tensor<Dtype>& lhs, const Tensor<Dtype>& rhs);



private:

	std::shared_ptr<Dtype> data_;
	int count_ = 0;
	// std::vector<int> shape_;

};

}

#endif //HYPERTEA_TENSOR_H_