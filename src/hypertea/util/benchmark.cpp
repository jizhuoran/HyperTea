#ifdef USE_BOOST
#include <boost/date_time/posix_time/posix_time.hpp>
#endif

#include "hypertea/common.hpp"
#include "hypertea/util/benchmark.hpp"

namespace hypertea {



#ifdef USE_OPENCL


void GPUTimer::Start() {
  if (!running()) {
    

    clReleaseEvent(start_gpu_cl_);

    cl_int ret;

    cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "null_kernel_float", &ret);
    OPENCL_CHECK(ret);

    int arg = 0;
    OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&arg));  

    size_t global_size = 1;

    OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &global_size, 0, NULL, &start_gpu_cl_));  
    clWaitForEvents(1, &start_gpu_cl_);
    
    clFinish(OpenCLHandler::Get().commandQueue);

    this->running_ = true;
    this->has_run_at_least_once_ = true;
  }
}

void GPUTimer::Stop() {
  if (running()) {
        
    clWaitForEvents(1, &stop_gpu_cl_);
    clReleaseEvent(stop_gpu_cl_);

    cl_int ret;

    cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "null_kernel_float", &ret);
    OPENCL_CHECK(ret);

    int arg = 0;
    OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&arg));  

    size_t global_size = 1;

    OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &global_size, 0, NULL, &stop_gpu_cl_));  
    
    clWaitForEvents(1, &stop_gpu_cl_);

    clFinish(OpenCLHandler::Get().commandQueue);
    running_ = false;
  }
}


float GPUTimer::MicroSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
      
  cl_ulong startTime = 0, stopTime = 0;
  OPENCL_CHECK(clGetEventProfilingInfo(start_gpu_cl_, CL_PROFILING_COMMAND_END,
      sizeof startTime, &startTime, NULL));
  OPENCL_CHECK(clGetEventProfilingInfo(stop_gpu_cl_, CL_PROFILING_COMMAND_START,
      sizeof stopTime, &stopTime, NULL));
  double ms = static_cast<double>(stopTime - startTime) / 1000.0;
  
  this->elapsed_microseconds_ = static_cast<float>(ms);
    
  return elapsed_microseconds_;
}

float GPUTimer::MilliSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
      
  cl_ulong startTime = 0, stopTime = 0;
  clGetEventProfilingInfo(start_gpu_cl_, CL_PROFILING_COMMAND_END,
      sizeof startTime, &startTime, NULL);
  clGetEventProfilingInfo(stop_gpu_cl_, CL_PROFILING_COMMAND_START,
      sizeof stopTime, &stopTime, NULL);
  double ms = static_cast<double>(stopTime - startTime) / 1000000.0;
  this->elapsed_milliseconds_ = static_cast<float>(ms);

  return elapsed_milliseconds_;
}


#endif




void CPUTimer::Start() {
  if (!running()) {
    gettimeofday(&start_cpu_, NULL);
    this->running_ = true;
    this->has_run_at_least_once_ = true;
  }
}

void CPUTimer::Stop() {
  if (running()) {
    gettimeofday(&stop_cpu_, NULL);
    this->running_ = false;
  }
}

float CPUTimer::MilliSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
    elapsed_milliseconds_ = (stop_cpu_.tv_sec - start_cpu_.tv_sec)*1000
    		+ (stop_cpu_.tv_usec - start_cpu_.tv_usec)/1000.0;
  return this->elapsed_milliseconds_;
}

float CPUTimer::MicroSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
    elapsed_microseconds_ = (stop_cpu_.tv_sec - start_cpu_.tv_sec)*1000000
    		+ (stop_cpu_.tv_usec - start_cpu_.tv_usec);
  return this->elapsed_microseconds_;
}

}  // namespace hypertea
