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
TensorCPU<Dtype>& inplace_gemm(
	const CBLAS_TRANSPOSE TransA,
	const CBLAS_TRANSPOSE TransB,
	const int M, const int N, const int K,
    const float alpha, 
    const TensorCPU<Dtype>& A, 
    const TensorCPU<Dtype>& B, 
    const float beta,
    TensorCPU<Dtype>& C) {

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
TensorCPU<Dtype>& inplace_gemv(
	const CBLAS_TRANSPOSE TransA, 
	const int M, const int N,
    const float alpha, 
    const TensorCPU<Dtype>& A, 
    const TensorCPU<Dtype>& x, 
    const float beta,
    TensorCPU<Dtype>& y) {

	auto A_data = A.immutable_data();
  	auto x_data = x.immutable_data();
  	auto y_data = y.mutable_data();

  	cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A_data, N, x_data, 1, beta, y_data, 1);

	return y;
}





template <typename Dtype>
inline TensorCPU<Dtype>& inplace_set(TensorCPU<Dtype> &x, const Dtype alpha) {
	
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
inline TensorCPU<Dtype>& inplace_sigmoid(TensorCPU<Dtype>& x) {
    
	vsSigmoid(x.count(), x.mutable_data());
	return x;
}


template <typename Dtype>
inline TensorCPU<Dtype>& inplace_tanh(TensorCPU<Dtype>& x) {
  
	vsTanH(x.count(), x.mutable_data());
	return x;
}



template <typename Dtype>
inline TensorCPU<Dtype>& inplace_abs(TensorCPU<Dtype>& x) {
	vsAbs(x.count(), x.mutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_exp(TensorCPU<Dtype>& x) {
	vsExp(x.count(), x.mutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_log(TensorCPU<Dtype>& x) {
	vsLn(x.count(), x.mutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_sqr(TensorCPU<Dtype>& x) {
	vsSqr(x.count(), x.mutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_sqrt(TensorCPU<Dtype>& x) {
	vsSqrt(x.count(), x.mutable_data());
	return x;
}


template <typename Dtype>
inline TensorCPU<Dtype>& inplace_inv(TensorCPU<Dtype>& x) {

	vsInv(x.count(), x.mutable_data());
	return x;
}


template <typename Dtype>
inline TensorCPU<Dtype>& inplace_powx(TensorCPU<Dtype>& x, const float a) {
	vsPowx(x.count(), a, x.mutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_elu(TensorCPU<Dtype>& x, const float a = 1.) {

	vsELU(x.count(), a, x.mutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_relu(TensorCPU<Dtype>& x, const float a = .0) {
	
	vsReLU(x.count(), a, x.mutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_add_scalar(TensorCPU<Dtype> &y, const float a) {

	vsAddScal(y.count(), a, y.mutable_data());
  	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_sub_scalar(TensorCPU<Dtype> &y, const float a) {

	vsAddScal(y.count(), -a, y.mutable_data());
  	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_mul_scalar(TensorCPU<Dtype> &y, const float a) {
	vsMulScal(y.count(), a, y.mutable_data());
  	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_div_scalar(TensorCPU<Dtype> &y, const float a) {
	
	vsMulScal(y.count(), 1/a, y.mutable_data());
  	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_add(const TensorCPU<Dtype>& x, TensorCPU<Dtype> &y) {
	vsAdd(y.count(), x.immutable_data(), y.mutable_data());
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_sub(const TensorCPU<Dtype>& x, TensorCPU<Dtype> &y) {
	vsSub(y.count(), x.immutable_data(), y.mutable_data());
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_mul(const TensorCPU<Dtype>& x, TensorCPU<Dtype> &y) {
	vsMul(y.count(), x.immutable_data(), y.mutable_data());
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_div(const TensorCPU<Dtype>& x, TensorCPU<Dtype> &y) {
	vsDiv(y.count(), x.immutable_data(), y.mutable_data());
	return y;
}








template <typename Dtype>
TensorCPU<Dtype>& inplace_prelu(
	TensorCPU<Dtype>& x, 
	const TensorCPU<Dtype>& weight,
	int channels,
	int spatial_dim
) {

	int num = x.count() / (channels * spatial_dim);

	auto data = x.mutable_data();
	auto weight_data = weight.immutable_data();

	for (auto& n: x.chunked_tensors(num)) {
		auto d = n.chunked_tensors(channels);

		for (int i = 0; i < channels; ++i) {
			inplace_relu(d[i], weight_data[i]);
		}
	}
	return x;
}


template <typename Dtype>
TensorCPU<Dtype>& inplace_channeled_scal(
	TensorCPU<Dtype>& x, 
	const TensorCPU<Dtype>& weight,
	int channels,
	int spatial_dim
) {

	int num = x.count() / (channels * spatial_dim);

	auto data = x.mutable_data();
	auto weight_data = weight.immutable_data();

	for (auto& n: x.chunked_tensors(num)) {
		auto d = n.chunked_tensors(channels);

		for (int i = 0; i < channels; ++i) {
			d[i] *= weight_data[i];
		}
	}
	return x;
}


template <typename Dtype>
TensorCPU<Dtype>& inplace_channeled_add(
	TensorCPU<Dtype>& x, 
	const TensorCPU<Dtype>& bias,
	int channels,
	int spatial_dim
) {

	int num = x.count() / (channels * spatial_dim);

	auto data = x.mutable_data();
	auto bias_data = bias.immutable_data();

	for (auto& n: x.chunked_tensors(num)) {
		auto d = n.chunked_tensors(channels);

		for (int i = 0; i < channels; ++i) {
			d[i] += bias_data[i];
		}
	}
	return x;
}


template <typename Dtype>
TensorCPU<Dtype>& inplace_channeled_sub(
	TensorCPU<Dtype>& x, 
	const TensorCPU<Dtype>& bias,
	int channels,
	int spatial_dim
) {

	int num = x.count() / (channels * spatial_dim);

	auto data = x.mutable_data();
	auto bias_data = bias.immutable_data();

	for (auto& n: x.chunked_tensors(num)) {
		auto d = n.chunked_tensors(channels);

		for (int i = 0; i < channels; ++i) {
			d[i] -= bias_data[i];
		}
	}
	return x;
}


template <typename Dtype>
TensorCPU<Dtype>& inplace_channeled_scaladd(
	TensorCPU<Dtype>& x, 
	const TensorCPU<Dtype>& weight,
	const TensorCPU<Dtype>& bias,
	int channels,
	int spatial_dim
) {
	int num = x.count() / (channels * spatial_dim);

	auto data = x.mutable_data();
	auto weight_data = weight.immutable_data();
	auto bias_data = bias.immutable_data();

	for (auto& n: x.chunked_tensors(num)) {
		auto d = n.chunked_tensors(channels);

		for (int i = 0; i < channels; ++i) {
			(d[i] *= weight_data[i]) += bias_data[i];
		}
	}

	return x;
}



template <typename Dtype>
TensorCPU<Dtype> outplace_gemv(
	const CBLAS_TRANSPOSE TransA, 
	const int M, const int N,
    const float alpha, 
    const TensorCPU<Dtype>& A, 
    const TensorCPU<Dtype>& x, 
    const float beta,
    const TensorCPU<Dtype>& y) {

	TensorCPU<Dtype> ny(y.count());
	ny.copy_data(y);

  	auto A_data = A.immutable_data();
  	auto x_data = x.immutable_data();
  	auto ny_data = ny.mutable_data();

  	cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A_data, N, x_data, 1, beta, ny_data, 1);

	return ny;
}

template <typename Dtype>
inline TensorCPU<Dtype> outplace_add(const TensorCPU<Dtype>& x, const TensorCPU<Dtype> &y) {
	auto z = x.duplicate();
	inplace_add(y, z);
	return z;
}

template <typename Dtype>
inline TensorCPU<Dtype> outplace_sub(const TensorCPU<Dtype>& x, const TensorCPU<Dtype> &y) {
	auto z = x.duplicate();
	inplace_sub(y, z);
	return z;
}

template <typename Dtype>
inline TensorCPU<Dtype> outplace_mul(const TensorCPU<Dtype>& x, const TensorCPU<Dtype> &y) {
	auto z = x.duplicate();
	inplace_mul(y, z);
	return z;
}

template <typename Dtype>
inline TensorCPU<Dtype> outplace_div(const TensorCPU<Dtype>& x, const TensorCPU<Dtype> &y) {
	auto z = x.duplicate();
	inplace_div(y, z);
	return z;
}

template <typename Dtype>
inline TensorCPU<Dtype> outplace_add_scalar(const TensorCPU<Dtype> &y, const float a) {
	auto z = y.duplicate();
	inplace_add_scalar(z, a);
	return z;
}

template <typename Dtype>
inline TensorCPU<Dtype> outplace_sub_scalar(const TensorCPU<Dtype> &y, const float a) {
	auto z = y.duplicate();
	inplace_sub_scalar(z, a);
	return z;
}

template <typename Dtype>
inline TensorCPU<Dtype> outplace_mul_scalar(const TensorCPU<Dtype> &y, const float a) {
	auto z = y.duplicate();
	inplace_mul_scalar(z, a);
	return z;
}

template <typename Dtype>
inline TensorCPU<Dtype> outplace_div_scalar(const TensorCPU<Dtype> &y, const float a) {
	auto z = y.duplicate();
	inplace_div_scalar(z, a);
	return z;
}


template <typename Dtype>
inline TensorCPU<Dtype> outplace_sigmoid(const TensorCPU<Dtype>& x) {
  	auto y = x.duplicate();
	inplace_sigmoid(y);
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype> outplace_tanh(const TensorCPU<Dtype>& x) {
  	auto y = x.duplicate();
	inplace_tanh(y);
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype> outplace_abs(const TensorCPU<Dtype>& x) {
	auto y = x.duplicate();
	inplace_abs(y);
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype> outplace_exp(const TensorCPU<Dtype>& x) {
	auto y = x.duplicate();
	inplace_exp(y);
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype> outplace_log(const TensorCPU<Dtype>& x) {
	auto y = x.duplicate();
	inplace_log(y);
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype> outplace_sqr(const TensorCPU<Dtype>& x) {
	auto y = x.duplicate();
	inplace_sqr(y);
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype> outplace_sqrt(const TensorCPU<Dtype>& x) {
	auto y = x.duplicate();
	inplace_sqrt(y);
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype> outplace_powx(const TensorCPU<Dtype>& x, const float a) {
	auto y = x.duplicate();
	inplace_powx(y, a);
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype> outplace_elu(const TensorCPU<Dtype>& x, const float a = 1.) {
	auto y = x.duplicate();
	inplace_elu(y, a);
	return y;
}


template <typename Dtype>
inline TensorCPU<Dtype> outplace_relu(const TensorCPU<Dtype>& x, const float a = .0) {
	auto y = x.duplicate();
	inplace_relu(y, a);
	return y;
}




template <typename Dtype>
void mean_var(
	TensorCPU<Dtype>& x, 
	TensorCPU<Dtype>& mean, TensorCPU<Dtype>& var, 
	int channels, int spatial_dim, float eps) {
	
	auto x_data = x.mutable_data();
	auto mean_data = mean.mutable_data();
	auto var_data = var.mutable_data();
	

	int nspatial_dim = x.count() / channels;

	Dtype p = 0;

	for (int c = 0; c < channels; ++c) {
		mean_data[c] = 0;
		var_data[c] = 0;
		for (int bs = 0; bs < (nspatial_dim / spatial_dim); ++bs) {
			for (int i = 0; i < spatial_dim; ++i) {
				p = x_data[bs * channels * spatial_dim + c * spatial_dim + i];
				mean_data[c] += p;
				var_data[c] += p*p;
			}
		}
		mean_data[c] /= nspatial_dim;
		var_data[c] /= nspatial_dim;
		var_data[c] = sqrt(var_data[c] - mean_data[c]*mean_data[c] + eps);
	}
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
TensorCPU<Dtype> upsampling_2d(
	TensorCPU<Dtype>& x,
	int scale,
	int height,
	int width,
	int spatial_dim) {


	int nums = x.count() / spatial_dim;

	TensorCPU<Dtype> y(x.count() * scale * scale);

	auto x_data = x.mutable_data();
	auto y_data = y.mutable_data();

	int index = 0;

	for (int n = 0; n < nums; ++n) {

		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {

				auto val = x_data[n * spatial_dim + i * width + j];

				for (int is = 0; is < scale; ++is) {
					for (int js = 0; js < scale; ++js) {
						y_data[n * spatial_dim * scale * scale + (i * scale + is) * width * scale + j * scale + js]= val;
					}				
				}		
			}
		}
	}

	return y;

}


template<typename Dtype> 
TensorCPU<Dtype> operator+ (const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs) {return outplace_add(lhs ,rhs); }
template<typename Dtype>
TensorCPU<Dtype> operator+ (const TensorCPU<Dtype>& lhs, const float rhs) {return outplace_add_scalar(lhs, rhs); }

template<typename Dtype>
TensorCPU<Dtype> operator- (const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs) {return outplace_sub(lhs ,rhs); }
template<typename Dtype>
TensorCPU<Dtype> operator- (const TensorCPU<Dtype>& lhs, const float rhs) {return outplace_sub_scalar(lhs, rhs); }

template<typename Dtype>
TensorCPU<Dtype> operator* (const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs) {return outplace_mul(lhs ,rhs); }
template<typename Dtype>
TensorCPU<Dtype> operator* (const TensorCPU<Dtype>& lhs, const float rhs) {return outplace_mul_scalar(lhs, rhs); }

template<typename Dtype>
TensorCPU<Dtype> operator/ (const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs) {return outplace_div(lhs ,rhs); }
template<typename Dtype> 
TensorCPU<Dtype> operator/ (const TensorCPU<Dtype>& lhs, const float rhs) {return outplace_div_scalar(lhs, rhs); }



}  // namespace hypertea

#endif  // HYPERTEA_UTIL_TENSOR_CPU_MATH_FUNC_H_
