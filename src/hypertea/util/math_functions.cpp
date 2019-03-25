#include <math.h>

#include <limits>

#include "hypertea/common.hpp"
#include "hypertea/util/math_functions.hpp"

#if defined(__APPLE__) && defined(__MACH__)
#include <vecLib.h>
#elif defined(USE_NEON_MATH) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace hypertea {


template<>
void hypertea_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}


template <>
void hypertea_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void hypertea_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }



#ifdef __VECLIB__
template <>
void hypertea_set(const int N, const float alpha, float* Y) {
  vDSP_vfill(&alpha, Y, 1, N);
}

#elif defined(__ARM_NEON_H)
template <>
void hypertea_set(const int N, const float alpha, float* Y) {
  int tail_frames = N % 4;
  const float* end = Y + N - tail_frames;
  while (Y < end) {
    float32x4_t alpha_dup = vld1q_dup_f32(&alpha);
    vst1q_f32(Y, alpha_dup);
    Y += 4;
  }
  for (int i = 0; i < tail_frames; ++i) {
    Y[i] = alpha;
  }
}

#else

template <>
void hypertea_set(const int N, const float alpha, float* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(float) * N);  // NOLINT(hypertea/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

#endif


void hypertea_set(const int N, const int alpha, int* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(int) * N);  // NOLINT(hypertea/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}


template <>
void hypertea_add_scalar(const int N, const float alpha, float* Y) {
#ifdef __VECLIB__
  vDSP_vsadd(Y, 1, &alpha, Y, 1, N);
#elif defined(__ARM_NEON_H)
  int tail_frames = N % 4;
  const float* end = Y + N - tail_frames;
  while (Y < end) {
    float32x4_t a_frame = vld1q_f32(Y);
    float32x4_t alpha_dup = vld1q_dup_f32(&alpha);
    vst1q_f32(Y, vaddq_f32(a_frame, alpha_dup));
    Y += 4;
  }
  for (int i = 0; i < tail_frames; ++i) {
    Y[i] += alpha;
  }
#else
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
#endif
}



template <typename Dtype>
void hypertea_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    //Different with HYPERTEA
    
    memcpy(Y, X, sizeof(Dtype) * N);
  }
}

template void hypertea_copy<int>(const int N, const int* X, int* Y);
template void hypertea_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);



#if defined(__ARM_NEON_H)
template <>
void hypertea_copy<float>(const int N, const float* X, float* Y) {
  int tail_frames = N % 4;
  const float* end = Y + N - tail_frames;
  while (Y < end) {
    float32x4_t x_frame = vld1q_f32(X);
    vst1q_f32(Y, x_frame);
    X += 4;
    Y += 4;
  }
  for (int i = 0; i < tail_frames; ++i) {
    Y[i] = X[i];
  }
}
#else
template void hypertea_copy<float>(const int N, const float* X, float* Y);
#endif




template <>
void hypertea_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void hypertea_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}


template <>
void hypertea_add<float>(const int n, const float* a, const float* b,
    float* y) {
#ifdef __VECLIB__
  vDSP_vadd(a, 1, b, 1, y, 1, n);
#elif defined(__ARM_NEON_H)
  int tail_frames = n % 4;
  const float* end = y + n - tail_frames;
  while (y < end) {
    float32x4_t a_frame = vld1q_f32(a);
    float32x4_t b_frame = vld1q_f32(b);
    vst1q_f32(y, vaddq_f32(a_frame, b_frame));
    a += 4;
    b += 4;
    y += 4;
  }
  vsAdd(tail_frames, a, b, y);
#else
  vsAdd(n, a, b, y);
#endif
}


template <>
void hypertea_sub<float>(const int n, const float* a, const float* b,
    float* y) {
#ifdef __VECLIB__
  vDSP_vsub(a, 1, b, 1, y, 1, n);
#elif defined(__ARM_NEON_H)
  int tail_frames = n % 4;
  const float* end = y + n - tail_frames;
  while (y < end) {
    float32x4_t a_frame = vld1q_f32(a);
    float32x4_t b_frame = vld1q_f32(b);
    vst1q_f32(y, vsubq_f32(a_frame, b_frame));
    a += 4;
    b += 4;
    y += 4;
  }
  vsSub(tail_frames, a, b, y);
#else
  vsSub(n, a, b, y);
#endif
}


template <>
void hypertea_mul<float>(const int n, const float* a, const float* b,
    float* y) {
#ifdef __VECLIB__
  vDSP_vmul(a, 1, b, 1, y, 1, n);
#elif defined(__ARM_NEON_H)
  int tail_frames = n % 4;
  const float* end = y + n - tail_frames;
  while (y < end) {
    float32x4_t a_frame = vld1q_f32(a);
    float32x4_t b_frame = vld1q_f32(b);
    vst1q_f32(y, vmulq_f32(a_frame, b_frame));
    a += 4;
    b += 4;
    y += 4;
  }
  vsMul(tail_frames, a, b, y);
#else
  vsMul(n, a, b, y);
#endif
}

template <>
void hypertea_div<float>(const int n, const float* a, const float* b,
    float* y) {
#ifdef __VECLIB__
  vDSP_vdiv(b, 1, a, 1, y, 1, n);
#elif defined(__ARM_NEON_H)
  int tail_frames = n % 4;
  const float* end = y + n - tail_frames;
  while (y < end) {
    float32x4_t a_frame = vld1q_f32(a);
    float32x4_t b_frame = vld1q_f32(b);
    vst1q_f32(y, vdivq_f32(a_frame, b_frame));
    a += 4;
    b += 4;
    y += 4;
  }
  vsDiv(tail_frames, a, b, y);
#else
  vsDiv(n, a, b, y);
#endif
}


template <>
void hypertea_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void hypertea_sqr<float>(const int n, const float* a, float* y) {
#ifdef __VECLIB__
  vDSP_vsq(a, 1, y, 1, n);
#elif defined(__ARM_NEON_H)
  int tail_frames = n % 4;
  const float* end = y + n - tail_frames;
  while (y < end) {
    float32x4_t a_frame = vld1q_f32(a);
    vst1q_f32(y, vmulq_f32(a_frame, a_frame));
    a += 4;
    y += 4;
  }
  vsSqr(tail_frames, a, y);
#else
  vsSqr(n, a, y);
#endif
}


template <>
void hypertea_sqrt<float>(const int n, const float* a, float* y) {
#ifdef __VECLIB__
  vvsqrtf(y, a, &n);
#elif defined(__ARM_NEON_H)
  int tail_frames = n % 4;
  const float* end = y + n - tail_frames;
  while (y < end) {
    float32x4_t a_frame = vld1q_f32(a);
    vst1q_f32(y, vsqrtq_f32(a_frame));
    a += 4;
    y += 4;
  }
  vsSqrt(tail_frames, a, y);
#else
  vsSqrt(n, a, y);
#endif
}


template <>
void hypertea_exp<float>(const int n, const float* a, float* y) {
#ifdef __VECLIB__
  vvexpf(y, a, &n);
#else
  vsExp(n, a, y);
#endif
}


template <>
void hypertea_log<float>(const int n, const float* a, float* y) {
#ifdef __VECLIB__
  vvlogf(y, a, &n);
#else
  vsLn(n, a, y);
#endif
}


template <>
void hypertea_abs<float>(const int n, const float* a, float* y) {
#ifdef __VECLIB__
  vDSP_vabs(a, 1, y , 1, n);
#elif defined(__ARM_NEON_H)
  int tail_frames = n % 4;
  const float* end = y + n - tail_frames;
  while (y < end) {
    float32x4_t a_frame = vld1q_f32(a);
    vst1q_f32(y, vabsq_f32(a_frame));
    a += 4;
    y += 4;
  }
  vsAbs(tail_frames, a, y);
#else
  vsAbs(n, a, y);
#endif
}


template <typename Dtype>
void hypertea_sigmoid(const int N, const Dtype* x, Dtype* y) {

  for (int i = 0; i < N; ++i) {
    y[i] = 0.5 * tanh(0.5 * x[i]) + 0.5;
  }
}
template void hypertea_sigmoid<float>(const int N, const float* x, float* y);


template <typename Dtype>
void hypertea_tanh(const int N, const Dtype* x, Dtype* y) {

  for (int i = 0; i < N; ++i) {
    y[i] = tanh(x[i]);
  }
}
template void hypertea_tanh<float>(const int N, const float* x, float* y);



template <>
float hypertea_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}


template <typename Dtype>
Dtype hypertea_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return hypertea_cpu_strided_dot(n, x, 1, y, 1);
}

template
float hypertea_cpu_dot<float>(const int n, const float* x, const float* y);



template <>
float hypertea_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}


template <>
void hypertea_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

}  // namespace hypertea
