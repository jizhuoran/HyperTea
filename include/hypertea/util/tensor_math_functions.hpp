#ifndef HYPERTEA_UTIL_TENSOR_MATH_FUNCTIONS_H_
#define HYPERTEA_UTIL_TENSOR_MATH_FUNCTIONS_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit



#include "hypertea/common.hpp"
#include "hypertea/util/device_alternate.hpp"
#include "hypertea/util/mkl_alternate.hpp"



namespace hypertea {


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


template <typename Dtype>
TensorGPU<Dtype> unary_math_gpu(
	const TensorGPU<Dtype> &x,
	const std::string& op_name
);


template <typename Dtype>
TensorGPU<Dtype> binary_math_gpu(
	const TensorGPU<Dtype> &x,
	const TensorGPU<Dtype> &y,
	const std::string& op_name
);


template <typename Dtype>
TensorGPU<Dtype>& unary_math_gpu_inplace(
	TensorGPU<Dtype> &x,
	const std::string& op_name
);


template <typename Dtype>
TensorGPU<Dtype>& binary_math_gpu_inplace(
	const TensorGPU<Dtype> &x,
	TensorGPU<Dtype> &y,
	const std::string& op_name
);


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
// TensorGPU<Dtype>& hypertea_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
//     const float alpha, const TensorGPU<Dtype>& A, const TensorGPU<Dtype>& x, const float beta,
//     TensorGPU<Dtype>& y);


// template <typename Dtype>
// void hypertea_gpu_bsum(const int m, const int n, const cl_mem X, const float alpha, const float beta,
//                             cl_mem y, const int x_inc);


// template <typename Dtype>
// void hypertea_gpu_axpy(const int N, const float alpha, const cl_mem X,
//     cl_mem Y);

// template <typename Dtype>
// void hypertea_gpu_axpby(const int N, const float alpha, const cl_mem X,
//     const float beta, cl_mem Y);

// void hypertea_gpu_memcpy(const size_t N, const void* X, void* Y);

// template <typename Dtype>
// void hypertea_gpu_set(const int N, const float alpha, cl_mem X);

// void hypertea_gpu_set(const int N, const int alpha, cl_mem X);

// inline void hypertea_gpu_memset(const size_t N, const int alpha, void* X) {

// #ifndef __ANDROID__ 
//   OPENCL_CHECK(clEnqueueFillBuffer(OpenCLHandler::Get().commandQueue, (cl_mem) X, &alpha, sizeof(int), 0, N, 0, NULL, NULL));
// #endif
// }

// template <typename Dtype>
// void hypertea_gpu_add_scalar(const int N, const float alpha, cl_mem X);

// template <typename Dtype>
// void hypertea_gpu_scal(const int N, const float alpha, cl_mem X);

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
inline TensorGPU<Dtype>& inplace_gpu_add_scalar(float a, TensorGPU<Dtype> &y) {
	return binary_math_gpu_inplace(x, y, "add_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype>& inplace_gpu_sub_scalar(float a, TensorGPU<Dtype> &y) {
	return binary_math_gpu_inplace(x, y, "sub_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype>& inplace_gpu_mul_scalar(float a, TensorGPU<Dtype> &y) {
	return binary_math_gpu_inplace(x, y, "mul_kernel");
}

template <typename Dtype>
inline TensorGPU<Dtype>& inplace_gpu_div_scalar(float a, TensorGPU<Dtype> &y) {
	return binary_math_gpu_inplace(x, y, "div_kernel");
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
TensorGPU<Dtype>& inplace_gpu_sigmoid(TensorGPU<Dtype>& x) {
  return unary_math_gpu_inplace(x, "SigmoidForward");
}

template <typename Dtype>
TensorGPU<Dtype>& inplace_gpu_tanh(TensorGPU<Dtype>& x) {
  return unary_math_gpu_inplace(x, "TanHForward");
}






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

// template <typename Dtype>
// void hypertea_gpu_abs(const int n, const cl_mem a, cl_mem y);

// template <typename Dtype>
// void hypertea_gpu_exp(const int n, const cl_mem a, cl_mem y);

// template <typename Dtype>
// void hypertea_gpu_log(const int n, const cl_mem a, cl_mem y);

// template <typename Dtype>
// void hypertea_gpu_powx(const int n, const cl_mem a, const float b, cl_mem y);

// template <typename Dtype>
// void hypertea_gpu_sqrt(const int n, const cl_mem a, cl_mem y);

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

#endif  // HYPERTEA_UTIL_TENSOR_MATH_FUNCTIONS_H_
