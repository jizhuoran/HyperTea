#include "hypertea/common.hpp"
#include "hypertea/util/benchmark.hpp"

namespace hypertea {

#ifdef USE_OPENCL

GPUTimer::GPUTimer()
    : initted_(false),
      running_(false),
      has_run_at_least_once_(false) {
  Init();
}

GPUTimer::~GPUTimer() { }


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

      running_ = true;
      has_run_at_least_once_ = true;
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
    LOG(WARNING) << "GPUTimer has never been run before reading time.";
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
  elapsed_microseconds_ = static_cast<float>(ms);

  return elapsed_microseconds_;
}

float GPUTimer::MilliSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "GPUTimer has never been run before reading time.";
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
  elapsed_milliseconds_ = static_cast<float>(ms);

  return elapsed_milliseconds_;
}

float GPUTimer::Seconds() {
  return MilliSeconds() / 1000.;
}

void GPUTimer::Init() {
  if (!initted()) {

    start_gpu_cl_ = 0;
    stop_gpu_cl_ = 0;

    initted_ = true;
  }
}

#endif //USE_OPENCL


CPUTimer::CPUTimer() {
  this->initted_ = true;
  this->running_ = false;
  this->has_run_at_least_once_ = false;
}

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


float CPUTimer::Seconds() {
  return MilliSeconds() / 1000.;
}

}  // namespace hypertea
