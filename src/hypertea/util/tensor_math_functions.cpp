 #include <math.h>

#include <limits>

#include "hypertea/common.hpp"
#include "hypertea/util/tensor_math_functions.hpp"

#ifdef USE_OPENCL

#include <clblast_c.h>
#include <clblast.h>

namespace hypertea {



template <typename Dtype>
TensorGPU<Dtype> unary_math_gpu(
  const TensorGPU<Dtype> &x,
  const std::string& op_name
) {

  int N = x.count();
  TensorGPU<Dtype> y(N);

  auto x_data = x.mutable_data();
  auto y_data = y.mutable_data();

  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    op_name,
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&x_data),
      std::make_pair(sizeof(cl_mem), (void *)&y_data),
      std::make_pair(sizeof(cl_int), (void *)&N)
    },
    std::vector<size_t> {HYPERTEA_GET_BLOCKS(N)},
    std::vector<size_t> {HYPERTEA_OPENCL_NUM_THREADS}
  );

  return y;

}

template TensorGPU<float> unary_math_gpu(
  const TensorGPU<float> &x,
  const std::string& op_name
);
template TensorGPU<half> unary_math_gpu(
  const TensorGPU<half> &x,
  const std::string& op_name
);



template <typename Dtype>
TensorGPU<Dtype> binary_math_gpu(
  const TensorGPU<Dtype> &x,
  const TensorGPU<Dtype> &y,
  const std::string& op_name
) {
  
  int N = x.count();
  TensorGPU<Dtype> z(N);

  cl_mem x_data = x.mutable_data();
  cl_mem y_data = y.mutable_data();
  cl_mem z_data = z.mutable_data();

  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    op_name,
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&x_data),
      std::make_pair(sizeof(cl_mem), (void *)&y_data),
      std::make_pair(sizeof(cl_mem), (void *)&z_data),
      std::make_pair(sizeof(cl_int), (void *)&N)
    },
    std::vector<size_t> {HYPERTEA_GET_BLOCKS(N)},
    std::vector<size_t> {HYPERTEA_OPENCL_NUM_THREADS}
  );

  return z;
}

template TensorGPU<float> binary_math_gpu(
  const TensorGPU<float> &x,
  const TensorGPU<float> &y,
  const std::string& op_name
);

template TensorGPU<half> binary_math_gpu(
  const TensorGPU<half> &x,
  const TensorGPU<half> &y,
  const std::string& op_name
);








template <typename Dtype>
TensorGPU<Dtype>& unary_math_gpu_inplace(
  TensorGPU<Dtype> &x,
  const std::string& op_name
) {

  int N = x.count();
  auto x_data = x.mutable_data();

  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    op_name,
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&x_data),
      std::make_pair(sizeof(cl_mem), (void *)&x_data),
      std::make_pair(sizeof(cl_int), (void *)&N)
    },
    std::vector<size_t> {HYPERTEA_GET_BLOCKS(N)},
    std::vector<size_t> {HYPERTEA_OPENCL_NUM_THREADS}
  );

  return x;

}

template TensorGPU<float>& unary_math_gpu_inplace(
  TensorGPU<float> &x,
  const std::string& op_name
);
template TensorGPU<half>& unary_math_gpu_inplace(
  TensorGPU<half> &x,
  const std::string& op_name
);



template <typename Dtype>
TensorGPU<Dtype>& binary_math_gpu_inplace(
  const TensorGPU<Dtype> &x,
  TensorGPU<Dtype> &y,
  const std::string& op_name
) {
  
  int N = x.count();
  cl_mem x_data = x.mutable_data();
  cl_mem y_data = y.mutable_data();

  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    op_name,
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&y_data),
      std::make_pair(sizeof(cl_mem), (void *)&x_data),
      std::make_pair(sizeof(cl_mem), (void *)&y_data),
      std::make_pair(sizeof(cl_int), (void *)&N)
    },
    std::vector<size_t> {HYPERTEA_GET_BLOCKS(N)},
    std::vector<size_t> {HYPERTEA_OPENCL_NUM_THREADS}
  );

  return y;
}

template TensorGPU<float>& binary_math_gpu_inplace(
  const TensorGPU<float> &x,
  TensorGPU<float> &y,
  const std::string& op_name
);

template TensorGPU<half>& binary_math_gpu_inplace(
  const TensorGPU<half> &x,
  TensorGPU<half> &y,
  const std::string& op_name
);

// template TensorGPU<float>& inplace_gpu_sigmoid<float>(TensorGPU<float>& x);
// template TensorGPU<half>&  inplace_gpu_sigmoid<half> (TensorGPU<half>&  x);



// template <typename Dtype>
// TensorGPU<Dtype>& inplace_gpu_tanh(TensorGPU<Dtype>& x) {
//   return unary_math_gpu(x, x, "TanHForward");
// }
// template TensorGPU<float>& inplace_gpu_tanh<float>(TensorGPU<float>& x);
// template TensorGPU<half>&  inplace_gpu_tanh<half> (TensorGPU<half>&  x);


template <typename Dtype>
TensorGPU<Dtype>& inplace_channeled_scal(

  TensorGPU<Dtype>& x, 
  const TensorGPU<Dtype>& weight,
  int channels,
  int inner_dim) {
  

  int N = x.count();
  auto data = x.mutable_data();
  auto weight_ = weight.mutable_data();


  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    "ScaleForward",
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&data),
      std::make_pair(sizeof(cl_mem), (void *)&data),
      std::make_pair(sizeof(cl_int), (void *)&N),
      std::make_pair(sizeof(cl_mem), (void *)&weight_),
      std::make_pair(sizeof(cl_int), (void *)&channels),
      std::make_pair(sizeof(cl_int), (void *)&inner_dim),

    },
    std::vector<size_t> {HYPERTEA_GET_BLOCKS(N)},
    std::vector<size_t> {HYPERTEA_OPENCL_NUM_THREADS}
  );

}

template TensorGPU<float>& inplace_channeled_scal(
  TensorGPU<float>& x, 
  const TensorGPU<float>& weight,
  int channels,
  int inner_dim
);

template TensorGPU<half>& inplace_channeled_scal(
  TensorGPU<half>& x, 
  const TensorGPU<half>& weight,
  int channels,
  int inner_dim
);



template <typename Dtype>
TensorGPU<Dtype>& inplace_channeled_add(

  TensorGPU<Dtype>& x, 
  const TensorGPU<Dtype>& bias,
  int channels,
  int inner_dim) {
  

  int N = x.count();
  auto data = x.mutable_data();
  auto bias_ = bias.mutable_data();


  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    "ChanneledAddForward",
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&data),
      std::make_pair(sizeof(cl_mem), (void *)&data),
      std::make_pair(sizeof(cl_int), (void *)&N),
      std::make_pair(sizeof(cl_mem), (void *)&bias_),
      std::make_pair(sizeof(cl_int), (void *)&channels),
      std::make_pair(sizeof(cl_int), (void *)&inner_dim),

    },
    std::vector<size_t> {HYPERTEA_GET_BLOCKS(N)},
    std::vector<size_t> {HYPERTEA_OPENCL_NUM_THREADS}
  );

}

template TensorGPU<float>& inplace_channeled_add(
  TensorGPU<float>& x, 
  const TensorGPU<float>& bias,
  int channels,
  int inner_dim
);

template TensorGPU<half>& inplace_channeled_add(
  TensorGPU<half>& x, 
  const TensorGPU<half>& bias,
  int channels,
  int inner_dim
);


template <typename Dtype>
TensorGPU<Dtype>& inplace_channeled_sub(
  TensorGPU<Dtype>& x, 
  const TensorGPU<Dtype>& bias,
  int channels,
  int inner_dim) {
  

  int N = x.count();
  auto data = x.mutable_data();
  auto bias_ = bias.mutable_data();


  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    "ChanneledSubForward",
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&data),
      std::make_pair(sizeof(cl_mem), (void *)&data),
      std::make_pair(sizeof(cl_int), (void *)&N),
      std::make_pair(sizeof(cl_mem), (void *)&bias_),
      std::make_pair(sizeof(cl_int), (void *)&channels),
      std::make_pair(sizeof(cl_int), (void *)&inner_dim),

    },
    std::vector<size_t> {HYPERTEA_GET_BLOCKS(N)},
    std::vector<size_t> {HYPERTEA_OPENCL_NUM_THREADS}
  );

}

template TensorGPU<float>& inplace_channeled_sub(
  TensorGPU<float>& x, 
  const TensorGPU<float>& bias,
  int channels,
  int inner_dim
);

template TensorGPU<half>& inplace_channeled_sub(
  TensorGPU<half>& x, 
  const TensorGPU<half>& bias,
  int channels,
  int inner_dim
);


template <typename Dtype>
TensorGPU<Dtype>& inplace_channeled_scaladd(
  TensorGPU<Dtype>& x, 
  const TensorGPU<Dtype>& weight,
  const TensorGPU<Dtype>& bias,
  int channels,
  int inner_dim) {
  

  int N = x.count();
  auto data = x.mutable_data();
  auto weight_ = weight.mutable_data();
  auto bias_ = bias.mutable_data();


  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    "ScaleBiasForward",
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&data),
      std::make_pair(sizeof(cl_mem), (void *)&data),
      std::make_pair(sizeof(cl_int), (void *)&N),
      std::make_pair(sizeof(cl_mem), (void *)&weight_),
      std::make_pair(sizeof(cl_mem), (void *)&bias_),
      std::make_pair(sizeof(cl_int), (void *)&channels),
      std::make_pair(sizeof(cl_int), (void *)&inner_dim),

    },
    std::vector<size_t> {HYPERTEA_GET_BLOCKS(N)},
    std::vector<size_t> {HYPERTEA_OPENCL_NUM_THREADS}
  );

}

template TensorGPU<float>& inplace_channeled_scaladd(
  TensorGPU<float>& x, 
  const TensorGPU<float>& weight,
  const TensorGPU<float>& bias,
  int channels,
  int inner_dim
);
template TensorGPU<half>& inplace_channeled_scaladd(
  TensorGPU<half>& x, 
  const TensorGPU<half>& weight,
  const TensorGPU<half>& bias,
  int channels,
  int inner_dim
);


}  // namespace hypertea

#endif //USE_OPENCL
