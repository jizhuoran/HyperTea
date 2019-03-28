#ifndef HYPERTEA_UTIL_MATH_FUNCTIONS_H_
#define HYPERTEA_UTIL_MATH_FUNCTIONS_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit



#include "hypertea/common.hpp"
#include "hypertea/util/device_alternate.hpp"
#include "hypertea/util/mkl_alternate.hpp"



namespace hypertea {

template<typename T> T to_dtype_(const float x);
template<> inline float to_dtype_<float>(const float x) {return x;}
template<> inline half to_dtype_<half> (const float x) {return float2half_impl(x);}

template<typename T> size_t dtype_size_();
template<> inline size_t dtype_size_<float>() {return sizeof(cl_float);}
template<> inline size_t dtype_size_<half> () {return sizeof(cl_half);}

// Hypertea gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
template <typename Dtype>
void hypertea_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const Dtype* A, const Dtype* B, const float beta,
    Dtype* C);

template <typename Dtype>
void hypertea_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const float alpha, const Dtype* A, const Dtype* x, const float beta,
    Dtype* y);


template <typename Dtype>
void hypertea_axpy(const int N, const float alpha, const Dtype* X,
    Dtype* Y);

template <typename Dtype>
void hypertea_cpu_axpby(const int N, const float alpha, const Dtype* X,
    const float beta, Dtype* Y);

template <typename Dtype>
void hypertea_copy(const int N, const Dtype *X, Dtype *Y);


template <typename Dtype>
void hypertea_set(const int N, const float alpha, Dtype *X);

void hypertea_set(const int N, const int alpha, int *X);

inline void hypertea_memset(const size_t N, const int alpha, void* X) {
  memset(X, alpha, N);  // NOLINT(hypertea/alt_fn)
}

template <typename Dtype>
void hypertea_add_scalar(const int N, const float alpha, Dtype *X);

template <typename Dtype>
void hypertea_scal(const int N, const float alpha, Dtype *X);

template <typename Dtype>
void hypertea_cpu_scale(const int n, const float alpha, const Dtype *x, Dtype* y);


template <typename Dtype>
void hypertea_sqr(const int N, const Dtype* a, Dtype* y);

template <typename Dtype>
void hypertea_sqrt(const int N, const Dtype* a, Dtype* y);

template <typename Dtype>
void hypertea_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void hypertea_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void hypertea_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void hypertea_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void hypertea_powx(const int n, const Dtype* a, const float b, Dtype* y);

unsigned int hypertea_rng_rand();

template <typename Dtype>
Dtype hypertea_nextafter(const Dtype b);

template <typename Dtype>
void hypertea_rng_uniform(const int n, const float a, const float b, Dtype* r);

template <typename Dtype>
void hypertea_rng_gaussian(const int n, const float mu, const float sigma,
                        Dtype* r);

template <typename Dtype>
void hypertea_rng_bernoulli(const int n, const Dtype p, int* r);

template <typename Dtype>
void hypertea_rng_bernoulli(const int n, const Dtype p, unsigned int* r);

template <typename Dtype>
void hypertea_exp(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void hypertea_log(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void hypertea_abs(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void hypertea_sigmoid(const int N, const Dtype* x, Dtype* y);

template <typename Dtype>
void hypertea_tanh(const int N, const Dtype* x, Dtype* y);


template <typename Dtype>
Dtype hypertea_cpu_dot(const int n, const Dtype* x, const Dtype* y);

template <typename Dtype>
Dtype hypertea_cpu_strided_dot(const int n, const Dtype* x, const int incx,
    const Dtype* y, const int incy);

// Returns the sum of the absolute values of the elements of vector x
template <typename Dtype>
Dtype hypertea_cpu_asum(const int n, const Dtype* x);

// the branchless, type-safe version from
// http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template<typename Dtype>
inline int8_t hypertea_sign(Dtype val) {
  return (Dtype(0) < val) - (val < Dtype(0));
}

// The following two macros are modifications of DEFINE_VSL_UNARY_FUNC
//   in include/hypertea/util/mkl_alternate.hpp authored by @Rowland Depp.
// Please refer to commit 7e8ef25c7 of the boost-eigen branch.
// Git cherry picking that commit caused a conflict hard to resolve and
//   copying that file in convenient for code reviewing.
// So they have to be pasted here temporarily.
#define DEFINE_HYPERTEA_CPU_UNARY_FUNC(name, operation) \
  template<typename Dtype> \
  void hypertea_cpu_##name(const int n, const Dtype* x, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(x); CHECK(y); \
    for (int i = 0; i < n; ++i) { \
      operation; \
    } \
  }

// output is 1 for the positives, 0 for zero, and -1 for the negatives
DEFINE_HYPERTEA_CPU_UNARY_FUNC(sign, y[i] = hypertea_sign<Dtype>(x[i]))

// This returns a nonzero value if the input has its sign bit set.
// The name sngbit is meant to avoid conflicts with std::signbit in the macro.
// The extra parens are needed because CUDA < 6.5 defines signbit as a macro,
// and we don't want that to expand here when CUDA headers are also included.
DEFINE_HYPERTEA_CPU_UNARY_FUNC(sgnbit, \
    y[i] = static_cast<bool>((std::signbit)(x[i])))

DEFINE_HYPERTEA_CPU_UNARY_FUNC(fabs, y[i] = std::fabs(x[i]))



#ifdef USE_OPENCL  // GPU

// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.


template <typename Dtype>
void hypertea_cl_copy(const int N, const cl_mem X, cl_mem Y, int x_offset = 0, int y_offset = 0);

template <typename Dtype>
void hypertea_gpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const cl_mem A, const cl_mem B, const float beta,
    cl_mem C);

template <typename Dtype>
void hypertea_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const float alpha, const cl_mem A, const cl_mem x, const float beta,
    cl_mem y);

template <typename Dtype>
void hypertea_gpu_bsum(const int m, const int n, const cl_mem X, const float alpha, const float beta,
                            cl_mem y, const int x_inc);


template <typename Dtype>
void hypertea_gpu_axpy(const int N, const float alpha, const cl_mem X,
    cl_mem Y);

template <typename Dtype>
void hypertea_gpu_axpby(const int N, const float alpha, const cl_mem X,
    const float beta, cl_mem Y);

void hypertea_gpu_memcpy(const size_t N, const void* X, void* Y);

template <typename Dtype>
void hypertea_gpu_set(const int N, const float alpha, cl_mem X);

void hypertea_gpu_set(const int N, const int alpha, cl_mem X);

inline void hypertea_gpu_memset(const size_t N, const int alpha, void* X) {

#ifndef __ANDROID__ 
  OPENCL_CHECK(clEnqueueFillBuffer(OpenCLHandler::Get().commandQueue, (cl_mem) X, &alpha, sizeof(int), 0, N, 0, NULL, NULL));
#endif
}

template <typename Dtype>
void hypertea_gpu_add_scalar(const int N, const float alpha, cl_mem X);


template <typename Dtype>
void hypertea_gpu_scal(const int N, const float alpha, cl_mem X);

template <typename Dtype>
void hypertea_gpu_add(const int N, const cl_mem a, const cl_mem b, cl_mem y);

template <typename Dtype>
void hypertea_gpu_sub(const int N, const cl_mem a, const cl_mem b, cl_mem y);

template <typename Dtype>
void hypertea_gpu_mul(const int N, const cl_mem a, const cl_mem b, cl_mem y);

template <typename Dtype>
void hypertea_gpu_div(const int N, const cl_mem a, const cl_mem b, cl_mem y);


template <typename Dtype>
void hypertea_gpu_sigmoid(const int N, const cl_mem x, cl_mem y);

template <typename Dtype>
void hypertea_gpu_tanh(const int N, const cl_mem x, cl_mem y);


template <typename Dtype>
void hypertea_gpu_abs(const int n, const cl_mem a, cl_mem y);

template <typename Dtype>
void hypertea_gpu_exp(const int n, const cl_mem a, cl_mem y);

template <typename Dtype>
void hypertea_gpu_log(const int n, const cl_mem a, cl_mem y);

template <typename Dtype>
void hypertea_gpu_powx(const int n, const cl_mem a, const float b, cl_mem y);

template <typename Dtype>
void hypertea_gpu_sqrt(const int n, const cl_mem a, cl_mem y);

// hypertea_gpu_rng_uniform with two arguments generates integers in the range
// [0, UINT_MAX].
void hypertea_gpu_rng_uniform(const int n, unsigned int* r);

// hypertea_gpu_rng_uniform with four arguments generates floats in the range
// (a, b] (strictly greater than a, less than or equal to b) due to the
// specification of curandGenerateUniform.  With a = 0, b = 1, just calls
// curandGenerateUniform; with other limits will shift and scale the outputs
// appropriately after calling curandGenerateUniform.
template <typename Dtype>
void hypertea_gpu_rng_uniform(const int n, const Dtype a, const Dtype b, cl_mem r);

template <typename Dtype>
void hypertea_gpu_rng_gaussian(const int n, const Dtype mu, const Dtype sigma,
                            cl_mem r);

template <typename Dtype>
void hypertea_gpu_rng_bernoulli(const int n, const Dtype p, int* r);

template <typename Dtype>
void hypertea_gpu_dot(const int n, const cl_mem x, const cl_mem y, cl_mem out, int x_offset = 0, int y_offset = 0);

template <typename Dtype>
void hypertea_gpu_asum(const int n, const cl_mem x, cl_mem y, int x_offset = 0);

template<typename Dtype>
void hypertea_gpu_sign(const int n, const cl_mem x, cl_mem y);

template<typename Dtype>
void hypertea_gpu_sgnbit(const int n, const cl_mem x, cl_mem y);

template <typename Dtype>
void hypertea_gpu_fabs(const int n, const cl_mem x, cl_mem y);

template <typename Dtype>
void hypertea_gpu_scale(const int n, const float alpha, const cl_mem x, cl_mem y);


#endif  // USE_OPENCL

}  // namespace hypertea

#endif  // HYPERTEA_UTIL_MATH_FUNCTIONS_H_
