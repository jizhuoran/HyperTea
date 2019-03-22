#ifndef HYPERTEA_UTIL_DEVICE_ALTERNATE_H_
#define HYPERTEA_UTIL_DEVICE_ALTERNATE_H_
#include <cxxabi.h>

#ifndef __ANDROID__
#include <execinfo.h>
#endif


#ifdef USE_OPENCL

#ifdef __ANDROID__

#define OPENCL_CHECK(condition) \
  do { \
    cl_int error = condition; \
    if(error != CL_SUCCESS) { \
      LOG(ERROR) << "This is a error for OpenCL " << error; \
      exit(1); \
    } \
  } while (0)


#define CLBLAST_CHECK(condition) \
  do { \
    CLBlastStatusCode status = condition; \
    if(status != CLBlastSuccess) { \
      LOG(ERROR) << "This is a error for CLBlast " << status; \
      exit(1); \
    } \
  } while (0)

#else // LINUX


#define OPENCL_CHECK(condition) \
  do { \
    cl_int error = condition; \
    if(error != CL_SUCCESS) { \
      std::cerr << "This is a error for OpenCL "<< error << " in " << __LINE__ << " in " << __FILE__ << std::endl;\
      void *buffer[100];\
      int n = backtrace(buffer,10);\
      char **str = backtrace_symbols(buffer, n);\
      for (int i = 0; i < n; i++) {printf("%d:  %s\n", i, str[i]);}\
      exit(1); \
    } \
  } while (0)


#define CLBLAST_CHECK(condition) \
  do { \
    CLBlastStatusCode status = condition; \
    if(status != CL_SUCCESS) { \
      std::cerr << "This is a error for CLBlast "<< status << " in " << __LINE__ << " in " << __FILE__ << std::endl;\
      void *buffer[100];\
      int n = backtrace(buffer,10);\
      char **str = backtrace_symbols(buffer, n);\
      for (int i = 0; i < n; i++) {printf("%d:  %s\n", i, str[i]);}\
      exit(1); \
    } \
  } while (0)


#define CLBLAST_CPP_CHECK(condition) \
  do { \
    auto status = condition; \
    if(status != clblast::StatusCode::kSuccess) { \
      std::cerr << "This is a error for CLBlast "<< static_cast<int>(status) << " in " << __LINE__ << " in " << __FILE__ << std::endl;\
      void *buffer[100];\
      int n = backtrace(buffer,10);\
      char **str = backtrace_symbols(buffer, n);\
      for (int i = 0; i < n; i++) {printf("%d:  %s\n", i, str[i]);}\
      exit(1); \
    } \
  } while (0)

  
#endif //END ANDROID  
#endif //END OPENCL




#define NOT_IMPLEMENT LOG(FATAL) << "This function has not been implemented yet!"


#ifdef USE_OPENCL


#define TEMP_GPU(classname) \
template <typename Dtype> \
void classname<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NOT_IMPLEMENT; }

#define TEMP_GPU_FORWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { funcname##_##cpu(bottom, top); } \

#define TEMP_GPU_BACKWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { funcname##_##cpu(top, propagate_down, bottom); } \

namespace hypertea {

// OPENCL: use 128 threads per block
const size_t HYPERTEA_OPENCL_NUM_THREADS = 128;

// OPENCL: number of blocks for threads.
inline int HYPERTEA_GET_BLOCKS(const int N) {
  return (N + HYPERTEA_OPENCL_NUM_THREADS - 1) / HYPERTEA_OPENCL_NUM_THREADS * HYPERTEA_OPENCL_NUM_THREADS;
}

}  // namespace hypertea

#else

#include <vector>

// Stub out GPU calls as unavailable.

#define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-only Hypertea: check mode."

#define STUB_GPU(classname) \
template <typename Dtype> \
void classname<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NO_GPU; }

#define STUB_CPU_FORWARD(classname) \
template <> \
void classname<half>::Forward_cpu(const half* bottom_data, \
    half* top_data) { LOG(FATAL) << "No Half on CPU"; }


#define STUB_GPU_FORWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NO_GPU; }

#define STUB_GPU_BACKWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { NO_GPU; }


#endif



#endif  // HYPERTEA_UTIL_DEVICE_ALTERNATE_H_
