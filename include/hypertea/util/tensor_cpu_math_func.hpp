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



// template <typename Dtype>
// TensorCPU<Dtype>& inplace_cpu_gemv(
// 	const CBLAS_TRANSPOSE TransA, 
// 	const int M, const int N,
//     const float alpha, 
//     const TensorCPU<Dtype>& A, 
//     const TensorCPU<Dtype>& x, 
//     const float beta,
//     TensorCPU<Dtype>& y) {

// 	Dtype alpha_(to_dtype<Dtype>(alpha));
//   	Dtype beta_(to_dtype<Dtype>(beta));

// 	auto A_data = A.immutable_data();
//   	auto x_data = x.immutable_data();
//   	auto y_data = y.immutable_data();


// 	auto blastTransA =
// 	(TransA != CblasNoTrans) ? clblast::Transpose::kNo : clblast::Transpose::kYes;

// 	CLBLAST_CPP_CHECK(clblast::Gemv<Dtype>(
// 		clblast::Layout::kColMajor,
// 		blastTransA, 
// 		N, M,
// 		alpha_,
// 		A_data, 0, N,
// 		x_data, 0, 1,
// 		beta_,
// 		y_data, 0, 1,
// 		&OpenCLHandler::Get().commandQueue, NULL)
// 	);

// 	return y;
// }


// template <typename Dtype>
// inline TensorCPU<Dtype>& inplace_cpu_set(TensorCPU<Dtype> &x, const Dtype alpha) {
// 	size_t x_size = x.count() * sizeof(Dtype);
// 	auto x_data = x.mutable_data();
// 	OPENCL_CHECK(clEnqueueFillBuffer(OpenCLHandler::Get().commandQueue, x_data, &alpha, sizeof(Dtype), 0, x_size, 0, NULL, NULL));
// 	return x;
// }


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










// template <typename Dtype>
// TensorCPU<Dtype> cpu_gemv(
// 	const CBLAS_TRANSPOSE TransA, 
// 	const int M, const int N,
//     const float alpha, 
//     const TensorCPU<Dtype>& A, 
//     const TensorCPU<Dtype>& x, 
//     const float beta,
//     const TensorCPU<Dtype>& y) {

// 	TensorCPU<Dtype> ny(y.count());
// 	ny.copy_data(y);

// 	Dtype alpha_(to_dtype<Dtype>(alpha));
//   	Dtype beta_(to_dtype<Dtype>(beta));

//   	auto A_data = A.immutable_data();
//   	auto x_data = x.immutable_data();
//   	auto ny_data = ny.immutable_data();


// 	auto blastTransA =
// 	(TransA != CblasNoTrans) ? clblast::Transpose::kNo : clblast::Transpose::kYes;

// 	CLBLAST_CPP_CHECK(clblast::Gemv<Dtype>(
// 		clblast::Layout::kColMajor,
// 		blastTransA, 
// 		N, M,
// 		alpha_,
// 		A_data, 0, N,
// 		x_data, 0, 1,
// 		beta_,
// 		ny_data, 0, 1,
// 		&OpenCLHandler::Get().commandQueue, NULL)
// 	);

// 	return ny;
// }

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

// template <typename Dtype>
// inline TensorCPU<Dtype>& inplace_cpu_sigmoid(TensorCPU<Dtype>& x) {
//   return unary_math_cpu_inplace(x, "SigmoidForward");
// }

// template <typename Dtype>
// inline TensorCPU<Dtype>& inplace_cpu_tanh(TensorCPU<Dtype>& x) {
//   return unary_math_cpu_inplace(x, "TanHForward");
// }

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_cpu_abs(TensorCPU<Dtype>& x) {
	vsAbs(x.count(), x.immutable_data(), x.mutable_data());
	return x;
}

// template <typename Dtype>
// inline TensorCPU<Dtype>& inplace_cpu_exp(TensorCPU<Dtype>& x) {
// 	return unary_math_cpu_inplace(x, "exp_kernel");
// }

template <typename Dtype>
inline TensorCPU<Dtype>& inplace_cpu_log(TensorCPU<Dtype>& x) {
	vsLn(x.count(), x.immutable_data(), x.mutable_data());
	return x;
}

// template <typename Dtype>
// inline TensorCPU<Dtype>& inplace_cpu_sqr(TensorCPU<Dtype>& x) {
// 	return unary_math_cpu_inplace(x, "sqr_kernel");
// }

// template <typename Dtype>
// inline TensorCPU<Dtype>& inplace_cpu_sqrt(TensorCPU<Dtype>& x) {
// 	return unary_math_cpu_inplace(x, "sqrt_kernel");
// }

// template <typename Dtype>
// inline TensorCPU<Dtype>& inplace_cpu_powx(TensorCPU<Dtype>& x, const float a) {
// 	return unary_scalar_math_cpu_inplace(x, a, "powx_kernel");
// }

// template <typename Dtype>
// inline TensorCPU<Dtype>& inplace_cpu_elu(TensorCPU<Dtype>& x, const float a = 1.) {
// 	return unary_scalar_math_cpu_inplace(x, a, "ELUForward");
// }

// template <typename Dtype>
// inline TensorCPU<Dtype>& inplace_cpu_relu(TensorCPU<Dtype>& x, const float a = .0) {
// 	return unary_scalar_math_cpu_inplace(x, a, "ReLUForward");
// }




// template <typename Dtype>
// inline TensorCPU<Dtype> cpu_sigmoid(const TensorCPU<Dtype>& x) {
//   return unary_math_cpu(x, "SigmoidForward");
// }

// template <typename Dtype>
// inline TensorCPU<Dtype> cpu_tanh(const TensorCPU<Dtype>& x) {
//   return unary_math_cpu(x, "TanHForward");
// }

template <typename Dtype>
inline TensorCPU<Dtype> cpu_abs(const TensorCPU<Dtype>& x) {
	auto y = x.duplicate();
	inplace_cpu_abs(y);
	return y;
}

// template <typename Dtype>
// inline TensorCPU<Dtype> cpu_exp(const TensorCPU<Dtype>& x) {
// 	return unary_math_cpu(x, "exp_kernel");
// }

template <typename Dtype>
inline TensorCPU<Dtype> cpu_log(const TensorCPU<Dtype>& x) {
	auto y = x.duplicate();
	inplace_cpu_log(y);
	return y;
}

// template <typename Dtype>
// inline TensorCPU<Dtype> cpu_sqr(const TensorCPU<Dtype>& x) {
// 	return unary_math_cpu(x, "sqr_kernel");
// }

// template <typename Dtype>
// inline TensorCPU<Dtype> cpu_sqrt(const TensorCPU<Dtype>& x) {
// 	return unary_math_cpu(x, "sqrt_kernel");
// }

// template <typename Dtype>
// inline TensorCPU<Dtype> cpu_powx(const TensorCPU<Dtype>& x, const float a) {
// 	return unary_scalar_math_cpu(x, a, "powx_kernel");
// }

// template <typename Dtype>
// inline TensorCPU<Dtype> cpu_elu(const TensorCPU<Dtype>& x, const float a = 1.) {
// 	return unary_scalar_math_cpu(x, a, "ELUForward");
// }


// template <typename Dtype>
// inline TensorCPU<Dtype> cpu_relu(const TensorCPU<Dtype>& x, const float a = .0) {
// 	return unary_scalar_math_cpu(x, a, "ReLUForward");
// }



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
