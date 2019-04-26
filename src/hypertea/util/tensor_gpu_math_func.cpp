#ifdef USE_OPENCL

#include <math.h>

#include <limits>

#include "hypertea/common.hpp"
#include "hypertea/util/tensor_gpu_math_func.hpp"


#include <clblast_c.h>
#include <clblast.h>

namespace hypertea {

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









template <typename Dtype>
TensorGPU<Dtype>& unary_scalar_math_gpu_inplace(
  TensorGPU<Dtype> &x,
  const float scalar,
  const std::string& op_name
) {

  int N = x.count();
  auto x_data = x.mutable_data();
  Dtype scalar_(to_dtype<Dtype>(scalar));

  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    op_name,
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&x_data),
      std::make_pair(sizeof(cl_mem), (void *)&x_data),
      std::make_pair(sizeof(Dtype), (void *)&scalar_),
      std::make_pair(sizeof(cl_int), (void *)&N)
    },
    std::vector<size_t> {HYPERTEA_GET_BLOCKS(N)},
    std::vector<size_t> {HYPERTEA_OPENCL_NUM_THREADS}
  );

  return x;

}

template TensorGPU<float>& unary_scalar_math_gpu_inplace(
  TensorGPU<float> &x,
  const float scalar,
  const std::string& op_name
);
template TensorGPU<half>& unary_scalar_math_gpu_inplace(
  TensorGPU<half> &x,
  const float scalar,
  const std::string& op_name
);



template <typename Dtype>
TensorGPU<Dtype>& binary_scalar_math_gpu_inplace(
  const TensorGPU<Dtype> &x,
  TensorGPU<Dtype> &y,
  const float scalar,
  const std::string& op_name
) {
  
  int N = x.count();
  cl_mem x_data = x.mutable_data();
  cl_mem y_data = y.mutable_data();
  Dtype scalar_(to_dtype<Dtype>(scalar));

  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    op_name,
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&y_data),
      std::make_pair(sizeof(cl_mem), (void *)&x_data),
      std::make_pair(sizeof(cl_mem), (void *)&y_data),
      std::make_pair(sizeof(Dtype), (void *)&scalar_),
      std::make_pair(sizeof(cl_int), (void *)&N)
    },
    std::vector<size_t> {HYPERTEA_GET_BLOCKS(N)},
    std::vector<size_t> {HYPERTEA_OPENCL_NUM_THREADS}
  );

  return y;
}

template TensorGPU<float>& binary_scalar_math_gpu_inplace(
  const TensorGPU<float> &x,
  TensorGPU<float> &y,
  const float scalar,
  const std::string& op_name
);

template TensorGPU<half>& binary_scalar_math_gpu_inplace(
  const TensorGPU<half> &x,
  TensorGPU<half> &y,
  const float scalar,
  const std::string& op_name
);





template <typename Dtype>
TensorGPU<Dtype> inplace_channeled_scal(

  TensorGPU<Dtype> x, 
  const TensorGPU<Dtype>& weight,
  int channels,
  int inner_dim) {
  

  int N = x.count();
  auto data = x.mutable_data();
  auto weight_ = weight.mutable_data();


  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    "channel_scal_kernel",
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

  return x;

}

template TensorGPU<float> inplace_channeled_scal(
  TensorGPU<float> x, 
  const TensorGPU<float>& weight,
  int channels,
  int inner_dim
);

template TensorGPU<half> inplace_channeled_scal(
  TensorGPU<half> x, 
  const TensorGPU<half>& weight,
  int channels,
  int inner_dim
);



template <typename Dtype>
TensorGPU<Dtype> inplace_channeled_add(

  TensorGPU<Dtype> x, 
  const TensorGPU<Dtype>& bias,
  int channels,
  int inner_dim) {
  

  int N = x.count();
  auto data = x.mutable_data();
  auto bias_ = bias.mutable_data();


  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    "channel_add_kernel",
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

  return x;

}

template TensorGPU<float> inplace_channeled_add(
  TensorGPU<float> x, 
  const TensorGPU<float>& bias,
  int channels,
  int inner_dim
);

template TensorGPU<half> inplace_channeled_add(
  TensorGPU<half> x, 
  const TensorGPU<half>& bias,
  int channels,
  int inner_dim
);


template <typename Dtype>
TensorGPU<Dtype> inplace_channeled_sub(
  TensorGPU<Dtype> x, 
  const TensorGPU<Dtype>& bias,
  int channels,
  int inner_dim) {
  

  int N = x.count();
  auto data = x.mutable_data();
  auto bias_ = bias.mutable_data();


  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    "channel_sub_kernel",
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

  return x;


}

template TensorGPU<float> inplace_channeled_sub(
  TensorGPU<float> x, 
  const TensorGPU<float>& bias,
  int channels,
  int inner_dim
);

template TensorGPU<half> inplace_channeled_sub(
  TensorGPU<half> x, 
  const TensorGPU<half>& bias,
  int channels,
  int inner_dim
);


template <typename Dtype>
TensorGPU<Dtype> inplace_channeled_scaladd(
  TensorGPU<Dtype> x, 
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
    "channel_scaladd_kernel",
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

  return x;


}

template TensorGPU<float> inplace_channeled_scaladd(
  TensorGPU<float> x, 
  const TensorGPU<float>& weight,
  const TensorGPU<float>& bias,
  int channels,
  int inner_dim
);
template TensorGPU<half> inplace_channeled_scaladd(
  TensorGPU<half> x, 
  const TensorGPU<half>& weight,
  const TensorGPU<half>& bias,
  int channels,
  int inner_dim
);



template <typename Dtype>
TensorGPU<Dtype> channeled_sum(
  TensorGPU<Dtype> x, 
  int spatial_dim) {
  
  int nums = x.count() / spatial_dim;
  TensorGPU<Dtype> sum(nums);


  auto x_data = x.mutable_data();
  auto sum_data = sum.mutable_data();

  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    "channeled_sum_kernel",
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&x_data),
      std::make_pair(sizeof(cl_mem), (void *)&sum_data),
      std::make_pair(sizeof(cl_int), (void *)&spatial_dim)
    },
    std::vector<size_t> {128, static_cast<size_t>(nums), 1},
    std::vector<size_t> {128, 1, 1}
  );

  return sum;

}


template TensorGPU<float> channeled_sum(
  TensorGPU<float> x, 
  int spatial_dim
);

template TensorGPU<half> channeled_sum(
  TensorGPU<half> x, 
  int spatial_dim
);





template <typename Dtype>
std::vector<int> batched_argmax(
  TensorGPU<Dtype> x, 
  int spatial_dim) {


  size_t batch_size = static_cast<size_t>(x.count() / spatial_dim);

  auto data = x.mutable_data();
  
  TensorGPU<Dtype> max_value(batch_size);
  auto max_value_ = max_value.mutable_data();



  cl_mem max_index_ = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, batch_size * sizeof(int), NULL, NULL);


  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    "argmax_kernel",
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&data),
      std::make_pair(sizeof(cl_mem), (void *)&max_value_),
      std::make_pair(sizeof(cl_mem), (void *)&max_index_),
      std::make_pair(sizeof(cl_int), (void *)&spatial_dim),
    },
    std::vector<size_t> {128, batch_size, 1},
    std::vector<size_t> {128, 1, 1}
  );


  auto max_index = std::vector<int>(batch_size);

  OPENCL_CHECK(clEnqueueReadBuffer(OpenCLHandler::Get().commandQueue, max_index_, CL_TRUE, 0, batch_size * sizeof(int), max_index.data(), 0, NULL, NULL));

  clReleaseMemObject(max_index_);

  return max_index;

}

template std::vector<int> batched_argmax(
  TensorGPU<float> x, 
  int spatial_dim
);

template std::vector<int> batched_argmax(
  TensorGPU<half> x, 
  int spatial_dim
);




}  // namespace hypertea

#endif //USE_OPENCL
