#ifndef HYPERTEA_UTIL_TENSOR_CPU_MATH_FUNC_H_
#define HYPERTEA_UTIL_TENSOR_CPU_MATH_FUNC_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <cblas.h>
// #include "hypertea/util/device_alternate.hpp"
// #include "hypertea/util/mkl_alternate.hpp"



namespace hypertea {

template<typename Dtype> class Tensor;
template<typename Dtype> class TensorCPU;


// template <typename Dtype>
// TensorCPU<Dtype> unary_math_cpu(const TensorCPU<Dtype> &x, const std::string& op_name);

// template <typename Dtype>
// TensorCPU<Dtype> unary_scalar_math_cpu(const TensorCPU<Dtype> &x, const float scalar, const std::string& op_name);

// template <typename Dtype>
// TensorCPU<Dtype>& unary_math_cpu_inplace(TensorCPU<Dtype> &x, const std::string& op_name);

// template <typename Dtype>
// TensorCPU<Dtype>& unary_scalar_math_cpu_inplace(TensorCPU<Dtype> &x, const float scalar, const std::string& op_name);




// template <typename Dtype>
// TensorCPU<Dtype> binary_math_cpu(const TensorCPU<Dtype> &x, const TensorCPU<Dtype> &y, const std::string& op_name);

// template <typename Dtype>
// TensorCPU<Dtype>& binary_math_cpu_inplace(const TensorCPU<Dtype> &x, TensorCPU<Dtype> &y, const std::string& op_name);

// template <typename Dtype>
// TensorCPU<Dtype> binary_scalar_math_cpu(const TensorCPU<Dtype> &x, const TensorCPU<Dtype> &y, const float scalar, const std::string& op_name);

// template <typename Dtype>
// TensorCPU<Dtype>& binary_scalar_math_cpu_inplace(const TensorCPU<Dtype> &x, TensorCPU<Dtype> &y, const float scalar, const std::string& op_name);



template <typename Dtype>
TensorCPU<Dtype>& inplace_cpu_gemv(
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
inline TensorCPU<Dtype>& inplace_cpu_set(TensorCPU<Dtype> &x, const Dtype alpha) {
	
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
inline TensorCPU<Dtype>& inplace_cpu_add(const TensorCPU<Dtype>& x, TensorCPU<Dtype> &y) {
	vsAdd(y.count(), y.immutable_data(), x.immutable_data(), y.mutable_data());
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_cpu_sub(const TensorCPU<Dtype>& x, TensorCPU<Dtype> &y) {
	vsSub(y.count(), y.immutable_data(), x.immutable_data(), y.mutable_data());
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_cpu_mul(const TensorCPU<Dtype>& x, TensorCPU<Dtype> &y) {
	vsMul(y.count(), y.immutable_data(), x.immutable_data(), y.mutable_data());
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_cpu_div(const TensorCPU<Dtype>& x, TensorCPU<Dtype> &y) {
	vsDiv(y.count(), y.immutable_data(), x.immutable_data(), y.mutable_data());
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_cpu_add_scalar(TensorCPU<Dtype> &y, const float a) {
	Dtype* y_data = y.mutable_data();
	for (int i = 0; i < y.count(); ++i) {
    	y_data[i] += a;
  	}
  	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_cpu_sub_scalar(TensorCPU<Dtype> &y, const float a) {
	Dtype* y_data = y.mutable_data();
	for (int i = 0; i < y.count(); ++i) {
    	y_data[i] -= a;
  	}
  	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_cpu_mul_scalar(TensorCPU<Dtype> &y, const float a) {
	Dtype* y_data = y.mutable_data();
	for (int i = 0; i < y.count(); ++i) {
    	y_data[i] *= a;
  	}
  	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_cpu_div_scalar(TensorCPU<Dtype> &y, const float a) {
	
	const float a_ = 1/a;

	Dtype* y_data = y.mutable_data();
	for (int i = 0; i < y.count(); ++i) {
    	y_data[i] *= a_;
  	}
  	return y;
}


template <typename Dtype>
inline TensorCPU<Dtype>& inplace_cpu_sigmoid(TensorCPU<Dtype>& x) {
    
    Dtype* data = x.mutable_data();
	for (int i = 0; i < x.size(); ++i) {
		data[i] = 0.5 * tanh(0.5 * data[i]) + 0.5;
	}
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_cpu_tanh(TensorCPU<Dtype>& x) {
  
    Dtype* data = x.mutable_data();
	for (int i = 0; i < x.size(); ++i) {
		data[i] = tanh(data[i]);
	}
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_cpu_abs(TensorCPU<Dtype>& x) {
	vsAbs(x.count(), x.immutable_data(), x.mutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_cpu_exp(TensorCPU<Dtype>& x) {
	vsExp(x.count(), x.immutable_data(), x.mutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_cpu_log(TensorCPU<Dtype>& x) {
	vsLn(x.count(), x.immutable_data(), x.mutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_cpu_sqr(TensorCPU<Dtype>& x) {
	vsSqr(x.count(), x.immutable_data(), x.mutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_cpu_sqrt(TensorCPU<Dtype>& x) {
	vsSqrt(x.count(), x.immutable_data(), x.mutable_data());
	return x;
}


template <typename Dtype>
inline TensorCPU<Dtype>& inplace_inv(TensorCPU<Dtype>& x, const float eps = 1e-5) {
	Dtype* data = x.mutable_data();
	for (int i = 0; i < x.count(); ++i) {
		data[i] = 1 / (data[i] + eps);
	}
	return x;
}


template <typename Dtype>
inline TensorCPU<Dtype>& inplace_cpu_powx(TensorCPU<Dtype>& x, const float a) {
	vsPowx(x.count(), x.immutable_data(), a, x.mutable_data());
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_cpu_elu(TensorCPU<Dtype>& x, const float a = 1.) {

	Dtype* data = x.mutable_data();
	for (int i = 0; i < x.count(); ++i) {
		data[i] = std::max(data[i], float(0)) + a * (exp(std::min(data[i], float(0))) - float(1));
	}
	return x;
}

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_cpu_relu(TensorCPU<Dtype>& x, const float a = .0) {
	
	Dtype* data = x.mutable_data();
	for (int i = 0; i < x.count(); ++i) {
		data[i] = std::max(data[i], float(0)) + a * std::min(data[i], float(0));
	}
	return x;
}


template <typename Dtype>
void mean_var(TensorCPU<Dtype>& x, TensorCPU<Dtype>& mean, TensorCPU<Dtype>& var, int channels, int batch_size) {
	
	auto x_data = x.mutable_data();
	auto mean_data = mean.mutable_data();
	auto var_data = mean.mutable_data();
	
	auto spatial_dim = x.count() / (channels * batch_size);

	auto p = 0;

	for (int c = 0; c < channels; ++c) {
		for (int bs = 0; bs < batch_size; ++bs) {
			for (int i = 0; i < spatial_dim; ++i) {
				p = x_data[bs * channels * spatial_dim + c * spatial_dim + i];
				mean_data[c] += p;
				var_data[c] += p*p;
			}
		}
		mean_data[c] /= (spatial_dim * batch_size);
		var_data[c] = var_data[c] / (spatial_dim * batch_size) - mean_data[c];
	}
}



template <typename Dtype>
TensorCPU<Dtype>& inplace_channeled_scal(
	TensorCPU<Dtype>& x, 
	const TensorCPU<Dtype>& weight,
	int channels,
	int num
) {

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
	int num
) {

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
	int num
) {

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
	int num
) {

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
TensorCPU<Dtype> cpu_gemv(
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
inline TensorCPU<Dtype> cpu_add(const TensorCPU<Dtype>& x, const TensorCPU<Dtype> &y) {
	auto z = x.duplicate();
	inplace_cpu_add(y, z);
	return z;
}

template <typename Dtype>
inline TensorCPU<Dtype> cpu_sub(const TensorCPU<Dtype>& x, const TensorCPU<Dtype> &y) {
	auto z = x.duplicate();
	inplace_cpu_sub(y, z);
	return z;
}

template <typename Dtype>
inline TensorCPU<Dtype> cpu_mul(const TensorCPU<Dtype>& x, const TensorCPU<Dtype> &y) {
	auto z = x.duplicate();
	inplace_cpu_mul(y, z);
	return z;
}

template <typename Dtype>
inline TensorCPU<Dtype> cpu_div(const TensorCPU<Dtype>& x, const TensorCPU<Dtype> &y) {
	auto z = x.duplicate();
	inplace_cpu_div(y, z);
	return z;
}

template <typename Dtype>
inline TensorCPU<Dtype> cpu_add_scalar(const TensorCPU<Dtype> &y, const float a) {
	auto z = y.duplicate();
	inplace_cpu_add_scalar(z, a);
	return z;
}

template <typename Dtype>
inline TensorCPU<Dtype> cpu_sub_scalar(const TensorCPU<Dtype> &y, const float a) {
	auto z = y.duplicate();
	inplace_cpu_sub_scalar(z, a);
	return z;
}

template <typename Dtype>
inline TensorCPU<Dtype> cpu_mul_scalar(const TensorCPU<Dtype> &y, const float a) {
	auto z = y.duplicate();
	inplace_cpu_mul_scalar(z, a);
	return z;
}

template <typename Dtype>
inline TensorCPU<Dtype> cpu_div_scalar(const TensorCPU<Dtype> &y, const float a) {
	auto z = y.duplicate();
	inplace_cpu_div_scalar(z, a);
	return z;
}


template <typename Dtype>
inline TensorCPU<Dtype> cpu_sigmoid(const TensorCPU<Dtype>& x) {
  	auto y = x.duplicate();
	inplace_cpu_sigmoid(y);
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype> cpu_tanh(const TensorCPU<Dtype>& x) {
  	auto y = x.duplicate();
	inplace_cpu_tanh(y);
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype> cpu_abs(const TensorCPU<Dtype>& x) {
	auto y = x.duplicate();
	inplace_cpu_abs(y);
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype> cpu_exp(const TensorCPU<Dtype>& x) {
	auto y = x.duplicate();
	inplace_cpu_exp(y);
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype> cpu_log(const TensorCPU<Dtype>& x) {
	auto y = x.duplicate();
	inplace_cpu_log(y);
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype> cpu_sqr(const TensorCPU<Dtype>& x) {
	auto y = x.duplicate();
	inplace_cpu_sqr(y);
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype> cpu_sqrt(const TensorCPU<Dtype>& x) {
	auto y = x.duplicate();
	inplace_cpu_sqrt(y);
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype> cpu_powx(const TensorCPU<Dtype>& x, const float a) {
	auto y = x.duplicate();
	inplace_cpu_powx(y, a);
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype> cpu_elu(const TensorCPU<Dtype>& x, const float a = 1.) {
	auto y = x.duplicate();
	inplace_cpu_elu(y, a);
	return y;
}


template <typename Dtype>
inline TensorCPU<Dtype> cpu_relu(const TensorCPU<Dtype>& x, const float a = .0) {
	auto y = x.duplicate();
	inplace_cpu_relu(y, a);
	return y;
}



template <typename Dtype>
inline TensorCPU<Dtype> outplace_tanh(const TensorCPU<Dtype>& x) {
  	auto y = x.duplicate();
	inplace_cpu_tanh(y);
	return y;
}

template <typename Dtype>
inline TensorCPU<Dtype> outplace_elu(const TensorCPU<Dtype>& x, const float a = 1.) {
	auto y = x.duplicate();
	inplace_cpu_elu(y, a);
	return y;
}


template <typename Dtype>
inline TensorCPU<Dtype> outplace_relu(const TensorCPU<Dtype>& x, const float a = .0) {
	auto y = x.duplicate();
	inplace_cpu_relu(y, a);
	return y;
}



// template <typename Dtype>
// TensorCPU<Dtype>& inplace_channeled_scal(
// 	TensorCPU<Dtype>& x, 
// 	const TensorCPU<Dtype>& weight,
// 	int channels,
// 	int inner_dim
// );


// template <typename Dtype>
// TensorCPU<Dtype>& inplace_channeled_add(
// 	TensorCPU<Dtype>& x, 
// 	const TensorCPU<Dtype>& bias,
// 	int channels,
// 	int inner_dim
// );

// template <typename Dtype>
// TensorCPU<Dtype>& inplace_channeled_sub(
// 	TensorCPU<Dtype>& x, 
// 	const TensorCPU<Dtype>& bias,
// 	int channels,
// 	int inner_dim
// );

// template <typename Dtype>
// TensorCPU<Dtype>& inplace_channeled_scaladd(
// 	TensorCPU<Dtype>& x, 
// 	const TensorCPU<Dtype>& weight,
// 	const TensorCPU<Dtype>& bias,
// 	int channels,
// 	int inner_dim
// );



// template <typename Dtype>
// void cpu_channeled_avg(
//   const TensorCPU<Dtype>& x, 
//   TensorCPU<Dtype>& mean,
//   TensorCPU<Dtype>& var,
//   int batch_size,
//   int spatial_dim
//  );


 
template<typename Dtype> 
TensorCPU<Dtype> operator+ (const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs) {return cpu_add(lhs ,rhs); }
template<typename Dtype>
TensorCPU<Dtype> operator+ (const TensorCPU<Dtype>& lhs, const float rhs) {return cpu_add_scalar(lhs, rhs); }

template<typename Dtype>
TensorCPU<Dtype> operator- (const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs) {return cpu_sub(lhs ,rhs); }
template<typename Dtype>
TensorCPU<Dtype> operator- (const TensorCPU<Dtype>& lhs, const float rhs) {return cpu_sub_scalar(lhs, rhs); }

template<typename Dtype>
TensorCPU<Dtype> operator* (const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs) {return cpu_mul(lhs ,rhs); }
template<typename Dtype>
TensorCPU<Dtype> operator* (const TensorCPU<Dtype>& lhs, const float rhs) {return cpu_mul_scalar(lhs, rhs); }

template<typename Dtype>
TensorCPU<Dtype> operator/ (const TensorCPU<Dtype>& lhs, const TensorCPU<Dtype>& rhs) {return cpu_div(lhs ,rhs); }
template<typename Dtype> 
TensorCPU<Dtype> operator/ (const TensorCPU<Dtype>& lhs, const float rhs) {return cpu_div_scalar(lhs, rhs); }



}  // namespace hypertea

#endif  // HYPERTEA_UTIL_TENSOR_CPU_MATH_FUNC_H_
