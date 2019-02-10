#ifndef HYPERTEA_LAYER_H_
#define HYPERTEA_LAYER_H_

#include <algorithm>
#include <string>
#include <vector>

#include "hypertea/common.hpp"
#include "hypertea/util/math_functions.hpp"


namespace hypertea {

/**
 * @brief An interface for the units of computation which can be composed into a
 *        Net.
 *
 * Layer%s must implement a Forward function, in which they take their input
 * (bottom) Blob%s (if any) and compute their output Blob%s (if any).
 * They may also implement a Backward function, in which they compute the error
 * gradients with respect to their input Blob%s, given the error gradients with
 * their output Blob%s.
 */
template <typename Dtype>
class Functor {

 public:
  /**
   * You should not implement your own constructor. Any set up code should go
   * to SetUp(), where the dimensions of the bottom blobs are provided to the
   * layer.
   */
  explicit Functor() {}
  virtual ~Functor() {}
  
  const size_t gpu_dtype_size();
  Dtype to_dtype(const float in);

  virtual inline const char* type() const { return ""; }



};  // class Layer



template <typename Dtype>
class CPUFunctor : public Functor<Dtype> {

public:
  CPUFunctor() : Functor<Dtype>() {}
  ~CPUFunctor() {}


  virtual void Forward(const Dtype* bottom_data,
      Dtype* top_data) {}

  virtual void Forward(const std::vector<Dtype*> bottom_datas,
      Dtype* top_data) {}

  virtual void Forward(const Dtype* bottom_data,
      const std::vector<Dtype*> top_datas) {}


  virtual void Forward(const std::vector<Dtype*> bottom_datas,
      const std::vector<Dtype*> top_datas) {}
  
};

#ifdef USE_OPENCL

template <typename Dtype>
class GPUFunctor : public Functor<Dtype> {

public:
  GPUFunctor() : Functor<Dtype>() {}
  ~GPUFunctor() {}


    virtual void Forward(const cl_mem bottom_data,
      cl_mem top_data) {}

    virtual void Forward(const cl_mem bottom_data,
      const std::vector<cl_mem> top_datas) {}

    virtual void Forward(const std::vector<cl_mem> bottom_datas,
      const std::vector<cl_mem> top_datas) {}

    virtual void Forward(const std::vector<cl_mem> bottom_datas,
      cl_mem top_data) {}
  
};

#endif //USE_OPENCL

}  // namespace hypertea

#endif  // HYPERTEA_LAYER_H_
