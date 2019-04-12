#ifndef HYPERTEA_UTIL_DEVICE_ALTERNATE_H_
#define HYPERTEA_UTIL_DEVICE_ALTERNATE_H_
#include <cxxabi.h>

#ifndef __ANDROID__
#include <execinfo.h>
#endif


#ifdef USE_OPENCL


#define OPENCL_BUILD_CHECK(condition) \
  do { \
    cl_int error = condition; \
    if (ret != CL_SUCCESS) { \
      char *buff_erro; \
      cl_int errcode; \
      size_t build_log_len; \
      errcode = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len); \
      if (errcode) { \
        LOG(ERROR) << "clGetProgramBuildInfo failed at line " << __LINE__; \
        exit(-1); \
      } \
      buff_erro = (char *)malloc(build_log_len); \
      if (!buff_erro) { \
          printf("malloc failed at line %d\n", __LINE__); \
          exit(-2); \
      } \
      errcode = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, build_log_len, buff_erro, NULL); \
      if (errcode) { \
          LOG(ERROR) << "clGetProgramBuildInfo failed at line " << __LINE__; \
          exit(-3); \
      } \
      LOG(ERROR) << "Build log: " << buff_erro; \
      free(buff_erro); \
      LOG(ERROR) << "clBuildProgram failed"; \
      exit(EXIT_FAILURE); \
    } \
  } while(0)




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
