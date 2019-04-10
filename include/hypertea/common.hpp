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
#include "hypertea/util/half.hpp"
#include "hypertea/util/opencl_util.hpp"
#include "hypertea/tensor.hpp"



// Convert macro to string
#define STRINGIFY(m) #m
#define AS_STRING(m) STRINGIFY(m)

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

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



#define DEFINE_FORWARD_FUNC(classname) \
template TensorCPU<float> classname<TensorCPU<float>>::operator()(TensorCPU<float>& input); \
template TensorGPU<float> classname<TensorGPU<float>>::operator()(TensorGPU<float>& input); \
template TensorGPU<half> classname<TensorGPU<half>>::operator()(TensorGPU<half>& input)




#define INSTANTIATE_CLASS_CPU(classname) \
char gInstantiationGuard##classname; \
template class classname<float>
// template class classname<double>

#define INSTANTIATE_LAYER_GPU_FORWARD(classname) \
  template void classname<half>::Forward_gpu( \
      const std::vector<Blob<half>*>& bottom, \
      const std::vector<Blob<half>*>& top); \
  template void classname<float>::Forward_gpu( \
      const std::vector<Blob<float>*>& bottom, \
      const std::vector<Blob<float>*>& top)
  // template void classname<double>::Forward_gpu( \
  //     const std::vector<Blob<double>*>& bottom, \
  //     const std::vector<Blob<double>*>& top);


#define INSTANTIATE_LAYER_GPU_FUNCS(classname) \
  INSTANTIATE_LAYER_GPU_FORWARD(classname); \

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"


#endif  // HYPERTEA_COMMON_HPP_
