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


namespace hypertea {



}  // namespace hypertea




#endif  // HYPERTEA_UTIL_DEVICE_ALTERNATE_H_
