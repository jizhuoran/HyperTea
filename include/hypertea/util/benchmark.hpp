#ifndef HYPERTEA_UTIL_BENCHMARK_H_
#define HYPERTEA_UTIL_BENCHMARK_H_

#include <sys/time.h>

#include "hypertea/util/opencl_util.hpp"

namespace hypertea {

class Timer { 
 public:
  Timer() : initted_(false), running_(false), has_run_at_least_once_(false) {}
  virtual ~Timer() {}
  virtual void Start() = 0;
  virtual void Stop() = 0;
  virtual float MilliSeconds() = 0;
  virtual float MicroSeconds() = 0;
  float Seconds() {return MilliSeconds() / 1000.;}

  inline bool initted() { return initted_; }
  inline bool running() { return running_; }
  inline bool has_run_at_least_once() { return has_run_at_least_once_; }

 protected:

  virtual void Init() = 0;

  bool initted_;
  bool running_;
  bool has_run_at_least_once_;
  

  float elapsed_milliseconds_;
  float elapsed_microseconds_;
};

#ifdef USE_OPENCL
class GPUTimer : public Timer {
 public:
  explicit GPUTimer() : Timer() {Init();}
  virtual ~GPUTimer() {}

  virtual void Init() {start_gpu_cl_ = 0; stop_gpu_cl_ = 0; initted_ = true;}
  virtual void Start();
  virtual void Stop();
  virtual float MilliSeconds();
  virtual float MicroSeconds();

  cl_event start_gpu_cl_;
  cl_event stop_gpu_cl_;

};
#endif


class CPUTimer : public Timer {
 public:
  explicit CPUTimer() : Timer() {Init();}
  virtual ~CPUTimer() {}
  virtual void Init() {initted_ = true;}
  virtual void Start();
  virtual void Stop();
  virtual float MilliSeconds();
  virtual float MicroSeconds();

  struct timeval start_cpu_;
  struct timeval stop_cpu_;
};

}  // namespace hypertea

#endif   // HYPERTEA_UTIL_BENCHMARK_H_
