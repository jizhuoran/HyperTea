#ifndef HYPERTEA_UTIL_CPU_BLAS_HELPER_H_
#define HYPERTEA_UTIL_CPU_BLAS_HELPER_H_

#include <math.h>

// Functions that hypertea uses but are not present if MKL is not linked.

// A simple way to define the vsl unary functions. The operation should
// be in the form e.g. y[i] = sqrt(a[i])
#define DEFINE_VSL_UNARY_FUNC(name, operation) \
  inline float* vs##name( \
    const int n, float* y) { \
    for (int i = 0; i < n; ++i) { operation; } \
    return y;\
  }


DEFINE_VSL_UNARY_FUNC(Sqr, y[i] *= y[i])
DEFINE_VSL_UNARY_FUNC(Sqrt, y[i] = sqrt(y[i]))
DEFINE_VSL_UNARY_FUNC(Exp, y[i] = exp(y[i]))
DEFINE_VSL_UNARY_FUNC(Ln, y[i] = log(y[i]))
DEFINE_VSL_UNARY_FUNC(Abs, y[i] = fabs(y[i]))
DEFINE_VSL_UNARY_FUNC(Sigmoid, y[i] = (0.5 * tanh(0.5 * y[i]) + 0.5))
DEFINE_VSL_UNARY_FUNC(TanH, y[i] = tanh(y[i]))
DEFINE_VSL_UNARY_FUNC(Inv, y[i] = 1 / y[i])

// A simple way to define the vsl unary functions with singular parameter b.
// The operation should be in the form e.g. y[i] = pow(a[i], b)
#define DEFINE_VSL_UNARY_FUNC_WITH_PARAM(name, operation) \
  inline float* vs##name( \
    const int n, const float b, float* y) { \
    for (int i = 0; i < n; ++i) { operation; } \
    return y;\
  } 


DEFINE_VSL_UNARY_FUNC_WITH_PARAM(Powx, y[i] = pow(y[i], b))
DEFINE_VSL_UNARY_FUNC_WITH_PARAM(AddScal, y[i] += b)
DEFINE_VSL_UNARY_FUNC_WITH_PARAM(MulScal, y[i] *= b)
DEFINE_VSL_UNARY_FUNC_WITH_PARAM(ELU, y[i] = std::max(y[i], float(0)) + b * (exp(std::min(y[i], float(0))) - float(1)))
DEFINE_VSL_UNARY_FUNC_WITH_PARAM(ReLU, y[i] = std::max(y[i], float(0)) + b * std::min(y[i], float(0)))


// A simple way to define the vsl binary functions. The operation should
// be in the form e.g. y[i] = a[i] + b[i]
#define DEFINE_VSL_BINARY_FUNC(name, operation) \
  inline float* vs##name( \
    const int n, const float* a, float* y) { \
    for (int i = 0; i < n; ++i) { operation; } \
    return y;\
  }


DEFINE_VSL_BINARY_FUNC(Add, y[i] += a[i])
DEFINE_VSL_BINARY_FUNC(Sub, y[i] -= a[i])
DEFINE_VSL_BINARY_FUNC(Mul, y[i] *= a[i])
DEFINE_VSL_BINARY_FUNC(Div, y[i] /= a[i])


#endif  // HYPERTEA_UTIL_CPU_BLAS_HELPER_H_
