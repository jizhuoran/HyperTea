#ifndef HYPERTEA_UTIL_BENCHMARK_H_
#define HYPERTEA_UTIL_BENCHMARK_H_

#include <sys/time.h>
#include "hypertea/util/device_alternate.hpp"

namespace hypertea {

#ifdef USE_OPENCL

class GPUTimer { 
 public:
  GPUTimer();
  virtual ~GPUTimer();
  virtual void Start();
  virtual void Stop();
  virtual float MilliSeconds();
  virtual float MicroSeconds();
  virtual float Seconds();

  inline bool initted() { return initted_; }
  inline bool running() { return running_; }
  inline bool has_run_at_least_once() { return has_run_at_least_once_; }

 protected:
  void Init();

  bool initted_;
  bool running_;
  bool has_run_at_least_once_;

  cl_event start_gpu_cl_;
  cl_event stop_gpu_cl_;


  float elapsed_milliseconds_;
  float elapsed_microseconds_;
};

#endif //USE_OPENCL

class CPUTimer{
 public:
  explicit CPUTimer();
  virtual ~CPUTimer() {}
  virtual void Start();
  virtual void Stop();
  virtual float MilliSeconds();
  virtual float MicroSeconds();
  virtual float Seconds();

  inline bool initted() { return initted_; }
  inline bool running() { return running_; }
  inline bool has_run_at_least_once() { return has_run_at_least_once_; }

 private:

  bool initted_;
  bool running_;
  bool has_run_at_least_once_;

  struct timeval start_cpu_;
  struct timeval stop_cpu_;


  float elapsed_milliseconds_;
  float elapsed_microseconds_;
};

}  // namespace hypertea

#endif   // HYPERTEA_UTIL_BENCHMARK_H_
