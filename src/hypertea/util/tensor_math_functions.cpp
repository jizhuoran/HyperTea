 #include <math.h>

#include <limits>

#include "hypertea/common.hpp"
#include "hypertea/util/tensor_math_functions.hpp"

#ifdef USE_OPENCL

#include <clblast_c.h>
#include <clblast.h>

namespace hypertea {


template <typename Dtype>
TensorGPU<Dtype>& inplace_gpu_sigmoid(TensorGPU<Dtype>& x) {

  int N = x.count();
  cl_mem data = x.mutable_data();

  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    "SigmoidForward",
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&data),
      std::make_pair(sizeof(cl_mem), (void *)&data),
      std::make_pair(sizeof(cl_int), (void *)&N)
    },
    std::vector<size_t> {HYPERTEA_GET_BLOCKS(N)},
    std::vector<size_t> {HYPERTEA_OPENCL_NUM_THREADS}
  );

  return x;
}

template TensorGPU<float>& inplace_gpu_sigmoid<float>(TensorGPU<float>& x);
template TensorGPU<half>&  inplace_gpu_sigmoid<half> (TensorGPU<half>&  x);



template <typename Dtype>
TensorGPU<Dtype>& inplace_gpu_tanh(TensorGPU<Dtype>& x) {

  int N = x.count();
  cl_mem data = x.mutable_data();

  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    "TanHForward",
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&data),
      std::make_pair(sizeof(cl_mem), (void *)&data),
      std::make_pair(sizeof(cl_int), (void *)&N)
    },
    std::vector<size_t> {HYPERTEA_GET_BLOCKS(N)},
    std::vector<size_t> {HYPERTEA_OPENCL_NUM_THREADS}
  );

  return x;

}

template TensorGPU<float>& inplace_gpu_tanh<float>(TensorGPU<float>& x);
template TensorGPU<half>&  inplace_gpu_tanh<half> (TensorGPU<half>&  x);



}  // namespace hypertea

#endif //USE_OPENCL
