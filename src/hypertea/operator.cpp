


#include "hypertea/operator.hpp"

namespace hypertea {

#ifdef USE_OPENCL

template <>
const size_t Functor<half>::gpu_dtype_size() {
  return sizeof(cl_half);
}

template <>
const size_t Functor<float>::gpu_dtype_size() {
  return sizeof(cl_float);
}

#endif //USE_OPENCL

template <>
half Functor<half>::to_dtype(const float in) {
  return float2half_impl(in);
}

template <>
float Functor<float>::to_dtype(const float in) {
  return in;
}


}