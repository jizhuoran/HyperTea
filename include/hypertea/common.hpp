#ifndef HYPERTEA_COMMON_HPP_
#define HYPERTEA_COMMON_HPP_


#include <memory>
#include "hypertea/glog_wrapper.hpp"

#ifdef USE_OPENCL
#define CL_TARGET_OPENCL_VERSION 220
#endif

#include <climits>
#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>
#include <assert.h>

#ifdef USE_OPENCL
#include <CL/cl.h>
#endif


#include "hypertea/util/device_alternate.hpp"
#include "hypertea/util/benchmark.hpp"
#include "hypertea/util/half.hpp"
#include "hypertea/util/opencl_util.hpp"
#include "hypertea/tensor.hpp"


#define IN_PLACE true
#define NOT_IN_PLACE false


#ifdef USE_OPENCL
#define INSTANTIATE_CLASS_GPU(classname) \
  char gInstantiationGuard##classname; \
  template class classname<half>; \
  template class classname<float>
  // template class classname<double>
#else
#define INSTANTIATE_CLASS_GPU(classname) \
  char gInstantiationGuard##classname;
#endif


#ifdef USE_OPENCL
#define DEFINE_FORWARD_FUNC(classname) \
template TensorCPU<float> classname<TensorCPU<float>>::operator()(TensorCPU<float> input); \
template TensorGPU<float> classname<TensorGPU<float>>::operator()(TensorGPU<float> input); \
template TensorGPU<half> classname<TensorGPU<half>>::operator()(TensorGPU<half> input)
#else
#define DEFINE_FORWARD_FUNC(classname) \
template TensorCPU<float> classname<TensorCPU<float>>::operator()(TensorCPU<float> input);
#endif //USE_OPENCL






#endif  // HYPERTEA_COMMON_HPP_
