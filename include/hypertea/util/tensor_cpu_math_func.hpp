#ifndef HYPERTEA_UTIL_TENSOR_CPU_MATH_FUNC_H_
#define HYPERTEA_UTIL_TENSOR_CPU_MATH_FUNC_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <cblas.h>
#include "hypertea/util/cpu_blas_helper.hpp"


namespace hypertea {

template<typename Dtype> class Tensor;
template<typename Dtype> class TensorCPU;


template <typename Dtype>
TensorCPU<Dtype> inplace_gemm(
	const CBLAS_TRANSPOSE TransA,
	const CBLAS_TRANSPOSE TransB,
	const int M, const int N, const int K,
    const float alpha, 
    const TensorCPU<Dtype>& A, 
    const TensorCPU<Dtype>& B, 
    const float beta,
    TensorCPU<Dtype> C) {

	auto A_data = A.immutable_data();
  	auto B_data = B.immutable_data();
  	auto C_data = C.mutable_data();


  	int lda = (TransA == CblasNoTrans) ? K : M;
  	int ldb = (TransB == CblasNoTrans) ? N : K;
  	cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A_data, lda, B_data,
      ldb, beta, C_data, N);

	return C;
}

template <typename Dtype>
TensorCPU<Dtype> outplace_gemm(
	const CBLAS_TRANSPOSE TransA,
	const CBLAS_TRANSPOSE TransB,
	const int M, const int N, const int K,
    const float alpha, 
    const TensorCPU<Dtype>& A, 
    const TensorCPU<Dtype>& B, 
    const float beta,
    const TensorCPU<Dtype>* const C = nullptr) {

	auto nC = C == nullptr? TensorCPU<Dtype>(M*N, 0) : C->duplicate();

	auto A_data = A.immutable_data();
  	auto B_data = B.immutable_data();
  	auto C_data = nC.mutable_data();


  	int lda = (TransA == CblasNoTrans) ? K : M;
  	int ldb = (TransB == CblasNoTrans) ? N : K;
  	cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A_data, lda, B_data,
      ldb, beta, C_data, N);

	return nC;
}


template <typename Dtype>
TensorCPU<Dtype> inplace_gemv(
	const CBLAS_TRANSPOSE TransA, 
	const int M, const int N,
    const float alpha, 
    const TensorCPU<Dtype>& A, 
    const TensorCPU<Dtype>& x, 
    const float beta,
    TensorCPU<Dtype> y) {

	auto A_data = A.immutable_data();
  	auto x_data = x.immutable_data();
  	auto y_data = y.mutable_data();

  	cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A_data, N, x_data, 1, beta, y_data, 1);

	return y;
}


template <typename Dtype>
TensorCPU<Dtype> outplace_gemv(
	const CBLAS_TRANSPOSE TransA, 
	const int M, const int N,
    const float alpha, 
    const TensorCPU<Dtype>& A, 
    const TensorCPU<Dtype>& x, 
    const float beta,
    const TensorCPU<Dtype>* const y = nullptr) {

	auto ny = y == nullptr? TensorCPU<Dtype>(M*N, 0) : y->duplicate();
  	auto A_data = A.immutable_data();
  	auto x_data = x.immutable_data();
  	auto ny_data = ny.mutable_data();

  	cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A_data, N, x_data, 1, beta, ny_data, 1);

	return ny;
}


template <typename Dtype>
inline TensorCPU<Dtype> inplace_set(TensorCPU<Dtype> x, const Dtype alpha) {
	
    if (alpha == 0) {
    	memset(x.mutable_data(), alpha, sizeof(Dtype) * x.count());
  	} else {
  		Dtype* data = x.mutable_data();
  		for (int i = 0; i < x.count(); ++i) {
	    	data[i] = alpha;
	  	}
  	}
  	return x;
}


template <typename Dtype>
inline TensorCPU<Dtype> inplace_sigmoid(TensorCPU<Dtype> x) {
	vsSigmoid(x.count(), x.mutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype> inplace_tanh(TensorCPU<Dtype> x) {
	vsTanH(x.count(), x.mutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype> inplace_abs(TensorCPU<Dtype> x) {
	vsAbs(x.count(), x.mutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype> inplace_exp(TensorCPU<Dtype> x) {
	vsExp(x.count(), x.mutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype> inplace_log(TensorCPU<Dtype> x) {
	vsLn(x.count(), x.mutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype> inplace_sqr(TensorCPU<Dtype> x) {
	vsSqr(x.count(), x.mutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype> inplace_sqrt(TensorCPU<Dtype> x) {
	vsSqrt(x.count(), x.mutable_data());
	return x;
}


template <typename Dtype>
inline TensorCPU<Dtype> inplace_inv(TensorCPU<Dtype> x) {
	vsInv(x.count(), x.mutable_data());
	return x;
}


template <typename Dtype>
inline TensorCPU<Dtype> inplace_powx(TensorCPU<Dtype> x, const float a) {
	vsPowx(x.count(), a, x.mutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype> inplace_elu(TensorCPU<Dtype> x, const float a = 1.) {

	vsELU(x.count(), a, x.mutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype> inplace_relu(TensorCPU<Dtype> x, const float a = .0) {
	
	vsReLU(x.count(), a, x.mutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype> inplace_add_scalar(TensorCPU<Dtype> y, const float a) {

	vsAddScal(y.count(), a, y.mutable_data());
  	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype> inplace_sub_scalar(TensorCPU<Dtype> y, const float a) {

	vsAddScal(y.count(), -a, y.mutable_data());
  	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype> inplace_mul_scalar(TensorCPU<Dtype> y, const float a) {
	vsMulScal(y.count(), a, y.mutable_data());
  	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype> inplace_div_scalar(TensorCPU<Dtype> y, const float a) {
	
	vsMulScal(y.count(), 1/a, y.mutable_data());
  	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype> inplace_add(TensorCPU<Dtype> x, const TensorCPU<Dtype>& y) {
	vsAdd(x.count(), x.mutable_data(), y.immutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype> inplace_sub(TensorCPU<Dtype> x, const TensorCPU<Dtype>& y) {
	vsSub(x.count(), x.mutable_data(), y.immutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype> inplace_mul(TensorCPU<Dtype> x, const TensorCPU<Dtype>& y) {
	vsMul(x.count(), x.mutable_data(), y.immutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype> inplace_div(TensorCPU<Dtype> x, const TensorCPU<Dtype>& y) {
	vsDiv(x.count(), x.mutable_data(), y.immutable_data());
	return x;
}

template <typename Dtype>
TensorCPU<Dtype> inplace_channeled_scal(
	TensorCPU<Dtype> x, 
	const TensorCPU<Dtype>& weight,
	int channels, int spatial_dim
) {

	auto weight_data = weight.immutable_data();
	DEFINE_VSL_CHANNEL_FUNC(data[(n * channels + c) * spatial_dim + i] *= weight_data[c]);
	return x;

}


template <typename Dtype>
TensorCPU<Dtype> inplace_channeled_add(
	TensorCPU<Dtype> x, 
	const TensorCPU<Dtype>& bias,
	int channels, int spatial_dim
) {

	auto bias_data = bias.immutable_data();
	DEFINE_VSL_CHANNEL_FUNC(data[(n * channels + c) * spatial_dim + i] += bias_data[c]);
	return x;

}


template <typename Dtype>
TensorCPU<Dtype> inplace_channeled_sub(
	TensorCPU<Dtype> x, 
	const TensorCPU<Dtype>& bias,
	int channels, int spatial_dim
) {

	auto bias_data = bias.immutable_data();
	DEFINE_VSL_CHANNEL_FUNC(data[(n * channels + c) * spatial_dim + i] -= bias_data[c]);
	return x;

}


template <typename Dtype>
TensorCPU<Dtype> inplace_channeled_scaladd(
	TensorCPU<Dtype> x, 
	const TensorCPU<Dtype>& weight,
	const TensorCPU<Dtype>& bias,
	int channels, int spatial_dim
) {

	auto weight_data = weight.immutable_data();
	auto bias_data = bias.immutable_data();
	DEFINE_VSL_CHANNEL_FUNC((data[(n * channels + c) * spatial_dim + i] *= weight_data[c]) += bias_data[c]);
	return x;

}





template <typename Dtype>
TensorCPU<Dtype> channeled_sum(
	TensorCPU<Dtype>& x, 
	int spatial_dim) {
	
	int nums = x.count() / spatial_dim;
	TensorCPU<Dtype> sum(nums);

	auto x_data = x.mutable_data();
	auto sum_data = sum.mutable_data();

	for (int n = 0; n < nums; ++n) {
		sum_data[n * spatial_dim] = 0;
		for (int i = 0; i < spatial_dim; ++i) {
			sum_data[n * spatial_dim] += x_data[n * spatial_dim + i];
		}
	}

	return sum;

}


template <typename Dtype>
std::vector<int> batched_argmax(
	TensorCPU<Dtype>& x, 
	int spatial_dim) {
	
	int batch_size = x.count() / spatial_dim;

	auto x_data = x.mutable_data();
	auto max_index = std::vector<int>(batch_size);

	int index = 0;

	for (int n = 0; n < batch_size; ++n) {

		Dtype max_value = -std::numeric_limits<float>::max();
		max_index[n] = -1;

		for (int i = 0; i < spatial_dim; ++i) {
			index = n * spatial_dim + i;
			if(x_data[index] > max_value) {
				max_value = x_data[index];
				max_index[n] = i;
			}
		}

	}

	return max_index;

}



template <typename Dtype>
TensorCPU<Dtype> concate(std::vector<TensorCPU<Dtype>* > xs) {

	int total_count = 0;

	for (auto const&x: xs) { total_count += x->count(); }

	TensorCPU<Dtype> y(total_count);
	auto y_data = y.mutable_data();

	int pos = 0;
	for (auto const&x: xs) {
    	memcpy(y_data + pos, x->immutable_data(), x->count() * sizeof(Dtype));
		pos += x->count();
	}

	return y;
}


template <typename Dtype>
TensorCPU<Dtype> hconcate(std::vector<TensorCPU<Dtype>* > xs, int top_dim) {

	int total_count = 0;

	for (auto const&x: xs) { total_count += x->count(); }

	TensorCPU<Dtype> y(total_count);
	auto y_data = y.mutable_data();

	int pos = 0;

	for (int i = 0; i < top_dim; ++i) {
		for (auto const&x: xs) {
    		memcpy(y_data + pos, x->immutable_data() + x->count() / top_dim * i, x->count() / top_dim * sizeof(Dtype));
			pos += (x->count() / top_dim);
		}
	}

	return y;
}


template<typename Dtype> 
TensorCPU<Dtype> operator+ (const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs) {return inplace_add(lhs.duplicate(), rhs); }
template<typename Dtype>
TensorCPU<Dtype> operator+ (const TensorCPU<Dtype>& lhs, const float rhs) {return inplace_add_scalar(lhs.duplicate(), rhs); }
template<typename Dtype>
TensorCPU<Dtype> operator+ (const float lhs, const TensorCPU<Dtype>& rhs) {return inplace_add_scalar(rhs.duplicate(), lhs); }

template<typename Dtype>
TensorCPU<Dtype> operator- (const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs) {return inplace_sub(lhs.duplicate(), rhs); }
template<typename Dtype>
TensorCPU<Dtype> operator- (const TensorCPU<Dtype>& lhs, const float rhs) {return inplace_sub_scalar(lhs.duplicate(), rhs); }
template<typename Dtype>
TensorCPU<Dtype> operator- (const float lhs, const TensorCPU<Dtype>& rhs) {return inplace_add_scalar(rhs.duplicate(), lhs); }

template<typename Dtype>
TensorCPU<Dtype> operator* (const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs) {return inplace_mul(lhs.duplicate(), rhs); }
template<typename Dtype>
TensorCPU<Dtype> operator* (const TensorCPU<Dtype>& lhs, const float rhs) {return inplace_mul_scalar(lhs.duplicate(), rhs); }
template<typename Dtype>
TensorCPU<Dtype> operator* (const float lhs, const TensorCPU<Dtype>& rhs) {return inplace_mul_scalar(rhs.duplicate(), lhs); }



template<typename Dtype>
TensorCPU<Dtype> operator/ (const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs) {return inplace_div(lhs.duplicate(), rhs); }
template<typename Dtype> 
TensorCPU<Dtype> operator/ (const TensorCPU<Dtype>& lhs, const float rhs) {return inplace_div_scalar(lhs.duplicate(), rhs); }
// template<typename Dtype>
// TensorCPU<Dtype> operator/ (const float lhs, const TensorCPU<Dtype>& rhs) {return outplace_div_scalar(rhs, lhs); }



}  // namespace hypertea

#endif  // HYPERTEA_UTIL_TENSOR_CPU_MATH_FUNC_H_
