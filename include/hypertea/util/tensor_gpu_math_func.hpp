#ifndef HYPERTEA_UTIL_TENSOR_GPU_MATH_FUNC_H_
#define HYPERTEA_UTIL_TENSOR_GPU_MATH_FUNC_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <clblast.h>


// #include "hypertea/common.hpp"
#include "hypertea/util/device_alternate.hpp"
#include "hypertea/util/mkl_alternate.hpp"



namespace hypertea {

template<typename Dtype> class Tensor;
template<typename Dtype> class TensorCPU;
template<typename Dtype> class TensorGPU;

// // The following two macros are modifications of DEFINE_VSL_UNARY_FUNC
// //   in include/hypertea/util/mkl_alternate.hpp authored by @Rowland Depp.
// // Please refer to commit 7e8ef25c7 of the boost-eigen branch.
// // Git cherry picking that commit caused a conflict hard to resolve and
// //   copying that file in convenient for code reviewing.
// // So they have to be pasted here temporarily.
// #define DEFINE_HYPERTEA_CPU_UNARY_FUNC(name, operation) \
//   template<typename Dtype> \
//   void hypertea_cpu_##name(const int n, const Dtype* x, Dtype* y) { \
//     CHECK_GT(n, 0); CHECK(x); CHECK(y); \
//     for (int i = 0; i < n; ++i) { \
//       operation; \
//     } \
//   }

// // output is 1 for the positives, 0 for zero, and -1 for the negatives
// DEFINE_HYPERTEA_CPU_UNARY_FUNC(sign, y[i] = hypertea_sign<Dtype>(x[i]))

// // This returns a nonzero value if the input has its sign bit set.
// // The name sngbit is meant to avoid conflicts with std::signbit in the macro.
// // The extra parens are needed because CUDA < 6.5 defines signbit as a macro,
// // and we don't want that to expand here when CUDA headers are also included.
// DEFINE_HYPERTEA_CPU_UNARY_FUNC(sgnbit, \
//     y[i] = static_cast<bool>((std::signbit)(x[i])))

// DEFINE_HYPERTEA_CPU_UNARY_FUNC(fabs, y[i] = std::fabs(x[i]))



#ifdef USE_OPENCL  // GPU

template<typename T> T to_dtype(const float x);
template<> inline float to_dtype<float>(const float x) {return x;}
template<> inline half to_dtype<half> (const float x) {return float2half_impl(x);}



template <typename Dtype>
TensorGPU<Dtype> unary_math_gpu(const TensorGPU<Dtype> &x, const std::string& op_name);

template <typename Dtype>
TensorGPU<Dtype> unary_scalar_math_gpu(const TensorGPU<Dtype> &x, const float scalar, const std::string& op_name);

template <typename Dtype>
TensorGPU<Dtype>& unary_math_gpu_inplace(TensorGPU<Dtype> &x, const std::string& op_name);

template <typename Dtype>
TensorGPU<Dtype>& unary_scalar_math_gpu_inplace(TensorGPU<Dtype> &x, const float scalar, const std::string& op_name);




template <typename Dtype>
TensorGPU<Dtype> binary_math_gpu(const TensorGPU<Dtype> &x, const TensorGPU<Dtype> &y, const std::string& op_name);

template <typename Dtype>
TensorGPU<Dtype>& binary_math_gpu_inplace(const TensorGPU<Dtype> &x, TensorGPU<Dtype> &y, const std::string& op_name);

template <typename Dtype>
TensorGPU<Dtype> binary_scalar_math_gpu(const TensorGPU<Dtype> &x, const TensorGPU<Dtype> &y, const float scalar, const std::string& op_name);

template <typename Dtype>
TensorGPU<Dtype>& binary_scalar_math_gpu_inplace(const TensorGPU<Dtype> &x, TensorGPU<Dtype> &y, const float scalar, const std::string& op_name);


// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.


// template <typename Dtype>
// void hypertea_cl_copy(const int N, const cl_mem X, cl_mem Y, int x_offset = 0, int y_offset = 0);

// template <typename Dtype>
// void hypertea_gpu_gemm(const CBLAS_TRANSPOSE TransA,
//     const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
//     const float alpha, const cl_mem A, const cl_mem B, const float beta,
//     cl_mem C);

// template <typename Dtype>
// void hypertea_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
//     const float alpha, const cl_mem A, const cl_mem x, const float beta,
//     cl_mem y);




// template <typename Dtype>
// void hypertea_gpu_bsum(const int m, const int n, const cl_mem X, const float alpha, const float beta,
//                             cl_mem y, const int x_inc);


template <typename Dtype>
TensorGPU<Dtype>& inplace_gpu_gemv(
	const CBLAS_TRANSPOSE TransA, 
	const int M, const int N,
    const float alpha, 
    const TensorGPU<Dtype>& A, 
    const TensorGPU<Dtype>& x, 
    const float beta,
    TensorGPU<Dtype>& y) {

	Dtype alpha_(to_dtype<Dtype>(alpha));
  	Dtype beta_(to_dtype<Dtype>(beta));

	auto A_data = A.immutable_data();
  	auto x_data = x.immutable_data();
  	auto y_data = y.immutable_data();


	auto blastTransA =
	(TransA != CblasNoTrans) ? clblast::Transpose::kNo : clblast::Transpose::kYes;

	CLBLAST_CPP_CHECK(clblast::Gemv<Dtype>(
		clblast::Layout::kColMajor,
		blastTransA, 
		N, M,
		alpha_,
		A_data, 0, N,
		x_data, 0, 1,
		beta_,
		y_data, 0, 1,
		&OpenCLHandler::Get().commandQueue, NULL)
	);

	return y;
}


template <typename Dtype>
TensorGPU<Dtype>& inplace_gemm(
	const CBLAS_TRANSPOSE TransA,
	const CBLAS_TRANSPOSE TransB,
	const int M, const int N, const int K,
    const float alpha, 
    const TensorGPU<Dtype>& A, 
    const TensorGPU<Dtype>& B, 
    const float beta,
    TensorGPU<Dtype>& C) {

	size_t lda = (TransA == CblasNoTrans) ? K : M;
  	size_t ldb = (TransB == CblasNoTrans) ? N : K;
  	size_t ldc = N;

	Dtype alpha_(to_dtype<Dtype>(alpha));
  	Dtype beta_(to_dtype<Dtype>(beta));

	auto A_data = A.immutable_data();
  	auto B_data = B.immutable_data();
  	auto C_data = C.immutable_data();

	auto blastTransA =
      (TransA == CblasNoTrans) ? clblast::Transpose::kNo : clblast::Transpose::kYes;
  	auto blastTransB =
      (TransB == CblasNoTrans) ? clblast::Transpose::kNo : clblast::Transpose::kYes;

	CLBLAST_CPP_CHECK(clblast::Gemm<Dtype>(
    clblast::Layout::kRowMajor,
    blastTransA, blastTransB,
    M, N, K,
    alpha_,
    (cl_mem) A_data, 0, lda,
    (cl_mem) B_data, 0, ldb,
    beta_,
    (cl_mem) C_data, 0, ldc,
    &OpenCLHandler::Get().commandQueue, NULL)
  );

	return C;
}



template <typename Dtype>
TensorGPU<Dtype>& inplace_gemv(
	const CBLAS_TRANSPOSE TransA, 
	const int M, const int N,
    const float alpha, 
    const TensorGPU<Dtype>& A, 
    const TensorGPU<Dtype>& x, 
    const float beta,
    TensorGPU<Dtype>& y) {

	Dtype alpha_(to_dtype<Dtype>(alpha));
  	Dtype beta_(to_dtype<Dtype>(beta));

	auto A_data = A.immutable_data();
  	auto x_data = x.immutable_data();
  	auto y_data = y.immutable_data();


	auto blastTransA =
	(TransA != CblasNoTrans) ? clblast::Transpose::kNo : clblast::Transpose::kYes;

	CLBLAST_CPP_CHECK(clblast::Gemv<Dtype>(
		clblast::Layout::kColMajor,
		blastTransA, 
		N, M,
		alpha_,
		A_data, 0, N,
		x_data, 0, 1,
		beta_,
		y_data, 0, 1,
		&OpenCLHandler::Get().commandQueue, NULL)
	);

	return y;
}


template <typename Dtype>
inline TensorGPU<Dtype>& inplace_gpu_set(TensorGPU<Dtype> &x, const Dtype alpha) {
	size_t x_size = x.count() * sizeof(Dtype);
	auto x_data = x.mutable_data();
	OPENCL_CHECK(clEnqueueFillBuffer(OpenCLHandler::Get().commandQueue, x_data, &alpha, sizeof(Dtype), 0, x_size, 0, NULL, NULL));
	return x;
}


template <typename Dtype>
inline TensorGPU<Dtype>& inplace_gpu_add(const TensorGPU<Dtype>& x, TensorGPU<Dtype> &y) {
	return binary_math_gpu_inplace(x, y, "add_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype>& inplace_gpu_sub(const TensorGPU<Dtype>& x, TensorGPU<Dtype> &y) {
	return binary_math_gpu_inplace(x, y, "sub_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype>& inplace_gpu_mul(const TensorGPU<Dtype>& x, TensorGPU<Dtype> &y) {
	return binary_math_gpu_inplace(x, y, "mul_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype>& inplace_gpu_div(const TensorGPU<Dtype>& x, TensorGPU<Dtype> &y) {
	return binary_math_gpu_inplace(x, y, "div_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype>& inplace_gpu_add_scalar(TensorGPU<Dtype> &y, const float a) {
	return unary_scalar_math_gpu_inplace(y, a, "outplace_add_scalar_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype>& inplace_gpu_sub_scalar(TensorGPU<Dtype> &y, const float a) {
	return unary_scalar_math_gpu_inplace(y, static_cast<float>(-a), "outplace_add_scalar_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype>& inplace_gpu_mul_scalar(TensorGPU<Dtype> &y, const float a) {
	return unary_scalar_math_gpu_inplace(y, a, "outplace_scal_scalar_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype>& inplace_gpu_div_scalar(TensorGPU<Dtype> &y, const float a) {
	return unary_scalar_math_gpu_inplace(y, static_cast<float>(1/a), "outplace_scal_scalar_kernel");
}










template <typename Dtype>
TensorGPU<Dtype> gpu_gemv(
	const CBLAS_TRANSPOSE TransA, 
	const int M, const int N,
    const float alpha, 
    const TensorGPU<Dtype>& A, 
    const TensorGPU<Dtype>& x, 
    const float beta,
    const TensorGPU<Dtype>& y) {

	TensorGPU<Dtype> ny(y.count());
	ny.copy_data(y);

	Dtype alpha_(to_dtype<Dtype>(alpha));
  	Dtype beta_(to_dtype<Dtype>(beta));

  	auto A_data = A.immutable_data();
  	auto x_data = x.immutable_data();
  	auto ny_data = ny.immutable_data();


	auto blastTransA =
	(TransA != CblasNoTrans) ? clblast::Transpose::kNo : clblast::Transpose::kYes;

	CLBLAST_CPP_CHECK(clblast::Gemv<Dtype>(
		clblast::Layout::kColMajor,
		blastTransA, 
		N, M,
		alpha_,
		A_data, 0, N,
		x_data, 0, 1,
		beta_,
		ny_data, 0, 1,
		&OpenCLHandler::Get().commandQueue, NULL)
	);

	return ny;
}

template <typename Dtype>
inline TensorGPU<Dtype> gpu_add(const TensorGPU<Dtype>& x, const TensorGPU<Dtype> &y) {
	return binary_math_gpu(x, y, "add_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype> gpu_sub(const TensorGPU<Dtype>& x, const TensorGPU<Dtype> &y) {
	return binary_math_gpu(x, y, "sub_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype> gpu_mul(const TensorGPU<Dtype>& x, const TensorGPU<Dtype> &y) {
	return binary_math_gpu(x, y, "mul_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype> gpu_div(const TensorGPU<Dtype>& x, const TensorGPU<Dtype> &y) {
	return binary_math_gpu(x, y, "div_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype> gpu_add_scalar(const TensorGPU<Dtype> &y, const float a) {
	return unary_scalar_math_gpu(y, a, "outplace_add_scalar_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype> gpu_sub_scalar(const TensorGPU<Dtype> &y, const float a) {
	return unary_scalar_math_gpu(y, static_cast<float>(-a), "outplace_add_scalar_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype> gpu_mul_scalar(const TensorGPU<Dtype> &y, const float a) {
	return unary_scalar_math_gpu(y, a, "outplace_scal_scalar_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype> gpu_div_scalar(const TensorGPU<Dtype> &y, const float a) {
	return unary_scalar_math_gpu(y, static_cast<float>(1/a), "outplace_scal_scalar_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype>& inplace_gpu_sigmoid(TensorGPU<Dtype>& x) {
  return unary_math_gpu_inplace(x, "SigmoidForward");
}

template <typename Dtype>
inline TensorGPU<Dtype>& inplace_gpu_tanh(TensorGPU<Dtype>& x) {
  return unary_math_gpu_inplace(x, "TanHForward");
}


template <typename Dtype>
inline TensorGPU<Dtype>& inplace_sigmoid(TensorGPU<Dtype>& x) {
  return unary_math_gpu_inplace(x, "SigmoidForward");
}

template <typename Dtype>
inline TensorGPU<Dtype>& inplace_tanh(TensorGPU<Dtype>& x) {
  return unary_math_gpu_inplace(x, "TanHForward");
}

template <typename Dtype>
inline TensorGPU<Dtype>& inplace_gpu_abs(TensorGPU<Dtype>& x) {
	return unary_math_gpu_inplace(x, "abs_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype>& inplace_gpu_exp(TensorGPU<Dtype>& x) {
	return unary_math_gpu_inplace(x, "exp_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype>& inplace_gpu_log(TensorGPU<Dtype>& x) {
	return unary_math_gpu_inplace(x, "log_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype>& inplace_gpu_sqr(TensorGPU<Dtype>& x) {
	return unary_math_gpu_inplace(x, "sqr_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype>& inplace_gpu_sqrt(TensorGPU<Dtype>& x) {
	return unary_math_gpu_inplace(x, "sqrt_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype>& inplace_gpu_powx(TensorGPU<Dtype>& x, const float a) {
	return unary_scalar_math_gpu_inplace(x, a, "powx_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype>& inplace_gpu_elu(TensorGPU<Dtype>& x, const float a = 1.) {
	return unary_scalar_math_gpu_inplace(x, a, "ELUForward");
}

template <typename Dtype>
inline TensorGPU<Dtype>& inplace_gpu_relu(TensorGPU<Dtype>& x, const float a = .0) {
	return unary_scalar_math_gpu_inplace(x, a, "ReLUForward");
}




template <typename Dtype>
inline TensorGPU<Dtype> gpu_sigmoid(const TensorGPU<Dtype>& x) {
  return unary_math_gpu(x, "SigmoidForward");
}

template <typename Dtype>
inline TensorGPU<Dtype> gpu_tanh(const TensorGPU<Dtype>& x) {
  return unary_math_gpu(x, "TanHForward");
}

template <typename Dtype>
inline TensorGPU<Dtype> gpu_abs(const TensorGPU<Dtype>& x) {
	return unary_math_gpu(x, "abs_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype> gpu_exp(const TensorGPU<Dtype>& x) {
	return unary_math_gpu(x, "exp_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype> gpu_log(const TensorGPU<Dtype>& x) {
	return unary_math_gpu(x, "log_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype> gpu_sqr(const TensorGPU<Dtype>& x) {
	return unary_math_gpu(x, "sqr_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype> gpu_sqrt(const TensorGPU<Dtype>& x) {
	return unary_math_gpu(x, "sqrt_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype> gpu_powx(const TensorGPU<Dtype>& x, const float a) {
	return unary_scalar_math_gpu(x, a, "powx_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype> gpu_elu(const TensorGPU<Dtype>& x, const float a = 1.) {
	return unary_scalar_math_gpu(x, a, "ELUForward");
}


template <typename Dtype>
inline TensorGPU<Dtype> gpu_relu(const TensorGPU<Dtype>& x, const float a = .0) {
	return unary_scalar_math_gpu(x, a, "ReLUForward");
}



template <typename Dtype>
inline TensorGPU<Dtype>& inplace_inv(TensorGPU<Dtype>& x, const float eps = 1e-5) {
	return unary_scalar_math_gpu_inplace(x, eps, "InvForward");
}


template <typename Dtype>
inline TensorGPU<Dtype> outplace_tanh(const TensorGPU<Dtype>& x) {
  return unary_math_gpu(x, "TanHForward");
}

template <typename Dtype>
inline TensorGPU<Dtype> outplace_elu(const TensorGPU<Dtype>& x, const float a = 1.) {
	return unary_scalar_math_gpu(x, a, "ELUForward");
}

template <typename Dtype>
inline TensorGPU<Dtype> outplace_relu(const TensorGPU<Dtype>& x, const float a = .0) {
	return unary_scalar_math_gpu(x, a, "ReLUForward");
}


template <typename Dtype>
void mean_var(const TensorGPU<Dtype>& x, TensorGPU<Dtype>& mean, TensorGPU<Dtype>& var, int channels, int spatial_dim, float eps);


template <typename Dtype>
TensorGPU<Dtype>& inplace_channeled_scal(
	TensorGPU<Dtype>& x, 
	const TensorGPU<Dtype>& weight,
	int channels,
	int inner_dim
);


template <typename Dtype>
TensorGPU<Dtype>& inplace_channeled_add(
	TensorGPU<Dtype>& x, 
	const TensorGPU<Dtype>& bias,
	int channels,
	int inner_dim
);

template <typename Dtype>
TensorGPU<Dtype>& inplace_channeled_sub(
	TensorGPU<Dtype>& x, 
	const TensorGPU<Dtype>& bias,
	int channels,
	int inner_dim
);

template <typename Dtype>
TensorGPU<Dtype>& inplace_channeled_scaladd(
	TensorGPU<Dtype>& x, 
	const TensorGPU<Dtype>& weight,
	const TensorGPU<Dtype>& bias,
	int channels,
	int inner_dim
);



template <typename Dtype>
void gpu_channeled_avg(
  const TensorGPU<Dtype>& x, 
  TensorGPU<Dtype>& mean,
  TensorGPU<Dtype>& var,
  int batch_size,
  int spatial_dim
 );


 
template<typename Dtype> 
TensorGPU<Dtype> operator+ (const TensorGPU<Dtype>& lhs, const TensorGPU<Dtype>& rhs) {return gpu_add(lhs ,rhs); }
template<typename Dtype>
TensorGPU<Dtype> operator+ (const TensorGPU<Dtype>& lhs, const float rhs) {return gpu_add_scalar(lhs, rhs); }

template<typename Dtype>
TensorGPU<Dtype> operator- (const TensorGPU<Dtype>& lhs, const TensorGPU<Dtype>& rhs) {return gpu_sub(lhs ,rhs); }
template<typename Dtype>
TensorGPU<Dtype> operator- (const TensorGPU<Dtype>& lhs, const float rhs) {return gpu_sub_scalar(lhs, rhs); }

template<typename Dtype>
TensorGPU<Dtype> operator* (const TensorGPU<Dtype>& lhs, const TensorGPU<Dtype>& rhs) {return gpu_mul(lhs ,rhs); }
template<typename Dtype>
TensorGPU<Dtype> operator* (const TensorGPU<Dtype>& lhs, const float rhs) {return gpu_mul_scalar(lhs, rhs); }

template<typename Dtype>
TensorGPU<Dtype> operator/ (const TensorGPU<Dtype>& lhs, const TensorGPU<Dtype>& rhs) {return gpu_div(lhs ,rhs); }
template<typename Dtype> 
TensorGPU<Dtype> operator/ (const TensorGPU<Dtype>& lhs, const float rhs) {return gpu_div_scalar(lhs, rhs); }
// template <typename Dtype>
// void hypertea_gpu_powx(const int n, const cl_mem a, const float b, cl_mem y);



// void hypertea_gpu_rng_bernoulli(const int n, const Dtype p, int* r);

// template <typename Dtype>
// void hypertea_gpu_dot(const int n, const cl_mem x, const cl_mem y, cl_mem out, int x_offset = 0, int y_offset = 0);

// template <typename Dtype>
// void hypertea_gpu_asum(const int n, const cl_mem x, cl_mem y, int x_offset = 0);

// template<typename Dtype>
// void hypertea_gpu_sign(const int n, const cl_mem x, cl_mem y);

// template<typename Dtype>
// void hypertea_gpu_sgnbit(const int n, const cl_mem x, cl_mem y);

// template <typename Dtype>
// void hypertea_gpu_fabs(const int n, const cl_mem x, cl_mem y);

// template <typename Dtype>
// void hypertea_gpu_scale(const int n, const float alpha, const cl_mem x, cl_mem y);


#endif  // USE_OPENCL

}  // namespace hypertea

#endif  // HYPERTEA_UTIL_TENSOR_GPU_MATH_FUNC_H_
