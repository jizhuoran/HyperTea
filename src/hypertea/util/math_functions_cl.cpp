 #include <math.h>

#include <limits>

#include "hypertea/common.hpp"
#include "hypertea/util/math_functions.hpp"

#ifdef USE_OPENCL

#include <clblast_c.h>
#include <clblast.h>

namespace hypertea {





template <typename Dtype>
void hypertea_gpu_gemm(
  const CBLAS_TRANSPOSE TransA,
  const CBLAS_TRANSPOSE TransB, 
  const int M, const int N, const int K,
  const float alpha, 
  const cl_mem A, 
  const cl_mem B, 
  const float beta,
  cl_mem C) {
  

  size_t lda = (TransA == CblasNoTrans) ? K : M;
  size_t ldb = (TransB == CblasNoTrans) ? N : K;
  size_t ldc = N;

  Dtype alpha_(to_dtype_<Dtype>(alpha));
  Dtype beta_(to_dtype_<Dtype>(beta));

  auto blastTransA =
      (TransA == CblasNoTrans) ? clblast::Transpose::kNo : clblast::Transpose::kYes;
  auto blastTransB =
      (TransB == CblasNoTrans) ? clblast::Transpose::kNo : clblast::Transpose::kYes;

  CLBLAST_CPP_CHECK(clblast::Gemm<Dtype>(
    clblast::Layout::kRowMajor,
    blastTransA, blastTransB,
    M, N, K,
    alpha_,
    (cl_mem) A, 0, lda,
    (cl_mem) B, 0, ldb,
    beta_,
    (cl_mem) C, 0, ldc,
    &OpenCLHandler::Get().commandQueue, NULL)
  );

}

template void hypertea_gpu_gemm<float>(
  const CBLAS_TRANSPOSE TransA,
  const CBLAS_TRANSPOSE TransB, 
  const int M, const int N, const int K,
  const float alpha, 
  const cl_mem A, 
  const cl_mem B, 
  const float beta,
  cl_mem C
);

template void hypertea_gpu_gemm<half>(
  const CBLAS_TRANSPOSE TransA,
  const CBLAS_TRANSPOSE TransB, 
  const int M, const int N, const int K,
  const float alpha, 
  const cl_mem A, 
  const cl_mem B, 
  const float beta,
  cl_mem C
);


template <typename Dtype>
TensorGPU<Dtype>& hypertea_gpu_gemv(
  const CBLAS_TRANSPOSE TransA, 
  const int M, const int N,
  const float alpha, 
  const TensorGPU<Dtype>& A, 
  const TensorGPU<Dtype>& x, 
  const float beta,
  TensorGPU<Dtype>& y) {

  hypertea_gpu_gemv<Dtype>(
    TransA, 
    M, N, 
    alpha, 
    A.immutable_data(),
    x.immutable_data(),
    beta,
    y.mutable_data()
  );

  return y;

}

template TensorGPU<float>& hypertea_gpu_gemv<float>(
  const CBLAS_TRANSPOSE TransA, 
  const int M, const int N,
  const float alpha, 
  const TensorGPU<float>& A, 
  const TensorGPU<float>& x, 
  const float beta,
  TensorGPU<float>& y);

template TensorGPU<half>& hypertea_gpu_gemv<half>(
  const CBLAS_TRANSPOSE TransA, 
  const int M, const int N,
  const float alpha, 
  const TensorGPU<half>& A, 
  const TensorGPU<half>& x, 
  const float beta,
  TensorGPU<half>& y);



template <typename Dtype>
void hypertea_gpu_gemv(
  const CBLAS_TRANSPOSE TransA, 
  const int M, const int N,
  const float alpha, 
  const cl_mem A, 
  const cl_mem x, 
  const float beta,
  cl_mem y) {


  Dtype alpha_(to_dtype_<Dtype>(alpha));
  Dtype beta_(to_dtype_<Dtype>(beta));


  auto blastTransA =
    (TransA != CblasNoTrans) ? clblast::Transpose::kNo : clblast::Transpose::kYes;

  CLBLAST_CPP_CHECK(clblast::Gemv<Dtype>(
      clblast::Layout::kColMajor,
      blastTransA, 
      N, M,
      alpha_,
      (cl_mem) A, 0, N,
      (cl_mem) x, 0, 1,
      beta_,
      (cl_mem) y, 0, 1,
      &OpenCLHandler::Get().commandQueue, NULL)
  );

}

template void hypertea_gpu_gemv<float>(
  const CBLAS_TRANSPOSE TransA, 
  const int M, const int N,
  const float alpha, 
  const cl_mem A, 
  const cl_mem x, 
  const float beta,
  cl_mem y);

template void hypertea_gpu_gemv<half>(
  const CBLAS_TRANSPOSE TransA, 
  const int M, const int N,
  const float alpha, 
  const cl_mem A, 
  const cl_mem x, 
  const float beta,
  cl_mem y);


template <typename Dtype>
void hypertea_gpu_bsum(const int m, const int n, const cl_mem X, const float alpha, const float beta,
                            cl_mem y, const int x_inc) {

  cl_int ret;

  cl_kernel kernel1 = clCreateKernel(OpenCLHandler::Get().math_program, "Xasum", &ret);
  OPENCL_CHECK(ret);
  cl_kernel kernel2 = clCreateKernel(OpenCLHandler::Get().math_program, "XasumEpilogue", &ret);
  OPENCL_CHECK(ret);

  Dtype alpha_(to_dtype_<Dtype>(alpha));
  Dtype beta_(to_dtype_<Dtype>(beta));

  size_t temp_size = 2*64;

  cl_mem temp_buffer = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, m * temp_size * sizeof(dtype_size_<Dtype>()), NULL, NULL);

  OPENCL_CHECK(clSetKernelArg(kernel1, 0, sizeof(cl_int), (void *)&n));  
  OPENCL_CHECK(clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void *)&X));  
  OPENCL_CHECK(clSetKernelArg(kernel1, 2, sizeof(cl_int), (void *)&x_inc));  
  OPENCL_CHECK(clSetKernelArg(kernel1, 3, sizeof(cl_mem), (void *)&temp_buffer));
  OPENCL_CHECK(clSetKernelArg(kernel1, 4, dtype_size_<Dtype>(), (void *)&alpha_));  



  size_t* local_size = new size_t[2];
  local_size[0] = static_cast<size_t>(64);
  local_size[1] = static_cast<size_t>(1);

  size_t* global_size = new size_t[2];
  global_size[0] = static_cast<size_t>(temp_size * 64);
  global_size[1] = static_cast<size_t>(m);



  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel1, 2, NULL, global_size, local_size, 0, NULL, NULL));  


  OPENCL_CHECK(clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&temp_buffer));  
  OPENCL_CHECK(clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void *)&y));
  OPENCL_CHECK(clSetKernelArg(kernel2, 2, dtype_size_<Dtype>(), (void *)&beta_));

  global_size[0] = static_cast<size_t>(64);

  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel2, 2, NULL, global_size, local_size, 0, NULL, NULL));  
 
  // OPENCL_CHECK(clReleaseMemObject(temp_buffer));


}
template void hypertea_gpu_bsum<float>(
  const int m, const int n, 
  const cl_mem X, 
  const float alpha, 
  const float beta,
  cl_mem y, 
  const int x_inc
);

template void hypertea_gpu_bsum<half>(
  const int m, const int n, 
  const cl_mem X, 
  const float alpha, 
  const float beta,
  cl_mem y, 
  const int x_inc
);


template <typename Dtype>
void hypertea_gpu_axpy(
  const int N, 
  const float alpha,
  const cl_mem X,
  cl_mem Y) {
      
  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "axpy_kernel", &ret);
  OPENCL_CHECK(ret);

  Dtype alpha_(to_dtype_<Dtype>(alpha));

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&X));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&Y));
  OPENCL_CHECK(clSetKernelArg(kernel, 2, dtype_size_<Dtype>(), (void *)&alpha_));
  OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&N));  

  size_t global_size = HYPERTEA_GET_BLOCKS(N);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  
  
}

template void hypertea_gpu_axpy<float>(
  const int N, 
  const float alpha,
  const cl_mem X,
  cl_mem Y
);

template void hypertea_gpu_axpy<half>(
  const int N, 
  const float alpha,
  const cl_mem X,
  cl_mem Y
);



template <typename Dtype>
void hypertea_gpu_set(const int N, const float alpha, cl_mem Y) {
  Dtype alpha_(to_dtype_<Dtype>(alpha));
  OPENCL_CHECK(clEnqueueFillBuffer(OpenCLHandler::Get().commandQueue, (cl_mem) Y, &alpha_, dtype_size_<Dtype>(), 0, N * dtype_size_<Dtype>(), 0, NULL, NULL));
}

template void hypertea_gpu_set<float>(const int N, const float alpha, cl_mem Y);
template void hypertea_gpu_set<half>(const int N, const float alpha, cl_mem Y);

void hypertea_gpu_set(const int N, const int alpha, cl_mem Y) {
  OPENCL_CHECK(clEnqueueFillBuffer(OpenCLHandler::Get().commandQueue, (cl_mem) Y, &alpha, sizeof(int), 0, N * sizeof(int), 0, NULL, NULL));
}


template <typename Dtype>
void hypertea_gpu_add_scalar(
  const int N, 
  const float alpha, 
  cl_mem X) {
  
  Dtype alpha_(to_dtype_<Dtype>(alpha));

  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "add_scalar_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&X));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, dtype_size_<Dtype>(), (void *)&alpha_));
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&N));  

  size_t global_size = HYPERTEA_GET_BLOCKS(N);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  
  
}

template void hypertea_gpu_add_scalar<float>(
  const int N, 
  const float alpha, 
  cl_mem X
);

template void hypertea_gpu_add_scalar<half>(
  const int N, 
  const float alpha, 
  cl_mem X
);



template <typename Dtype>
void hypertea_gpu_scal(
  const int N, 
  const float alpha, 
  cl_mem X) {
  
  Dtype alpha_(to_dtype_<Dtype>(alpha));

  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    "scal_kernel",
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&X),
      std::make_pair(dtype_size_<Dtype>(), (void *)&alpha_),
      std::make_pair(sizeof(cl_int), (void *)&N)
    },
    std::vector<size_t> {HYPERTEA_GET_BLOCKS(N)},
    std::vector<size_t> {HYPERTEA_OPENCL_NUM_THREADS}
  );



}

template void hypertea_gpu_scal<float>(const int N, const float alpha, cl_mem X);
template void hypertea_gpu_scal<half>(const int N, const float alpha, cl_mem X);


template <typename Dtype>
void hypertea_gpu_axpby(
  const int N, 
  const float alpha, 
  const cl_mem X,
  const float beta, 
  cl_mem Y) {
    hypertea_gpu_scal<Dtype>(N, beta, Y);
    hypertea_gpu_axpy<Dtype>(N, alpha, X, Y);
}

template void hypertea_gpu_axpby<float>(
  const int N, 
  const float alpha, 
  const cl_mem X,
  const float beta, 
  cl_mem Y
);

template void hypertea_gpu_axpby<half>(
  const int N, 
  const float alpha, 
  const cl_mem X,
  const float beta, 
  cl_mem Y
);



template <typename Dtype>
void hypertea_gpu_add(
  const int N, 
  const cl_mem a, 
  const cl_mem b, 
  cl_mem y) {
    

  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    "add_kernel",
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&a),
      std::make_pair(sizeof(cl_mem), (void *)&b),
      std::make_pair(sizeof(cl_mem), (void *)&y),
      std::make_pair(sizeof(cl_int), (void *)&N)
    },
    std::vector<size_t> {HYPERTEA_GET_BLOCKS(N)},
    std::vector<size_t> {HYPERTEA_OPENCL_NUM_THREADS}
  );

}

template void hypertea_gpu_add<float>(
  const int N, 
  const cl_mem a, 
  const cl_mem b, 
  cl_mem y
);

template void hypertea_gpu_add<half>(
  const int N, 
  const cl_mem a, 
  const cl_mem b, 
  cl_mem y
);


template <typename Dtype>
cl_mem _hypertea_gpu_add(
  const int N, 
  const cl_mem a, 
  const cl_mem b, 
  cl_mem y) {
    

  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    "add_kernel",
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&a),
      std::make_pair(sizeof(cl_mem), (void *)&b),
      std::make_pair(sizeof(cl_mem), (void *)&y),
      std::make_pair(sizeof(cl_int), (void *)&N)
    },
    std::vector<size_t> {HYPERTEA_GET_BLOCKS(N)},
    std::vector<size_t> {HYPERTEA_OPENCL_NUM_THREADS}
  );

  return y;

}

template cl_mem _hypertea_gpu_add<float>(
  const int N, 
  const cl_mem a, 
  const cl_mem b, 
  cl_mem y
);

template cl_mem _hypertea_gpu_add<half>(
  const int N, 
  const cl_mem a, 
  const cl_mem b, 
  cl_mem y
);



template <typename Dtype>
void hypertea_gpu_sub(
  const int N, 
  const cl_mem a, 
  const cl_mem b, 
  cl_mem y) {
    
  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "sub_kernel", &ret);
  OPENCL_CHECK(ret); 

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&y));  
  OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&N));  

  size_t global_size = HYPERTEA_GET_BLOCKS(N);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  
  
}

template void hypertea_gpu_sub<float>(
  const int N, 
  const cl_mem a, 
  const cl_mem b, 
  cl_mem y
);

template void hypertea_gpu_sub<half>(
  const int N, 
  const cl_mem a, 
  const cl_mem b, 
  cl_mem y
);




template <typename Dtype>
void hypertea_gpu_mul(
  const int N, 
  const cl_mem a, 
  const cl_mem b, 
  cl_mem y) {
    
  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "mul_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&y));  
  OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&N));  

  size_t global_size = HYPERTEA_GET_BLOCKS(N);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  
  
}

template void hypertea_gpu_mul<float>(
  const int N, 
  const cl_mem a, 
  const cl_mem b, 
  cl_mem y
);

template void hypertea_gpu_mul<half>(
  const int N, 
  const cl_mem a, 
  const cl_mem b, 
  cl_mem y
);


template <typename Dtype>
void hypertea_gpu_div(
  const int N, 
  const cl_mem a, 
  const cl_mem b, 
  cl_mem y) {
    
  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "div_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&y));  
  OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&N));  

  size_t global_size = HYPERTEA_GET_BLOCKS(N);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  
  
}

template void hypertea_gpu_div<float>(
  const int N, 
  const cl_mem a, 
  const cl_mem b, 
  cl_mem y
);

template void hypertea_gpu_div<half>(
  const int N, 
  const cl_mem a, 
  const cl_mem b, 
  cl_mem y
);



template <typename Dtype>
void hypertea_gpu_sigmoid(
  const int N, 
  const cl_mem x, 
  cl_mem y) {

  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    "SigmoidForward",
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&x),
      std::make_pair(sizeof(cl_mem), (void *)&y),
      std::make_pair(sizeof(cl_int), (void *)&N)
    },
    std::vector<size_t> {HYPERTEA_GET_BLOCKS(N)},
    std::vector<size_t> {HYPERTEA_OPENCL_NUM_THREADS}
  );
}

template void hypertea_gpu_sigmoid<float>(
  const int N, 
  const cl_mem x, 
  cl_mem y
);

template void hypertea_gpu_sigmoid<half>(
  const int N, 
  const cl_mem x, 
  cl_mem y
);



template <typename Dtype>
void hypertea_gpu_tanh(const int N, const cl_mem x, cl_mem y) {
  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    "TanHForward",
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&x),
      std::make_pair(sizeof(cl_mem), (void *)&y),
      std::make_pair(sizeof(cl_int), (void *)&N)
    },
    std::vector<size_t> {HYPERTEA_GET_BLOCKS(N)},
    std::vector<size_t> {HYPERTEA_OPENCL_NUM_THREADS}
  );
}
template void hypertea_gpu_tanh<float>(
  const int N, 
  const cl_mem x, 
  cl_mem y
);

template void hypertea_gpu_tanh<half>(
  const int N, 
  const cl_mem x, 
  cl_mem y
);


template <typename Dtype>
void hypertea_gpu_abs(
  const int n, 
  const cl_mem a, 
  cl_mem y) {
  
  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "abs_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&y));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&n));  

  size_t global_size = HYPERTEA_GET_BLOCKS(n);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  
  
}

template void hypertea_gpu_abs<float>(
  const int n, 
  const cl_mem a, 
  cl_mem y
);

template void hypertea_gpu_abs<half>(
  const int n, 
  const cl_mem a, 
  cl_mem y
);




template <typename Dtype>
void hypertea_gpu_exp(
  const int n, 
  const cl_mem a, 
  cl_mem y) {
  
  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "exp_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&y));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&n));  

  size_t global_size = HYPERTEA_GET_BLOCKS(n);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  
  
}

template void hypertea_gpu_exp<float>(
  const int n, 
  const cl_mem a, 
  cl_mem y
);

template void hypertea_gpu_exp<half>(
  const int n, 
  const cl_mem a, 
  cl_mem y
);





template <typename Dtype>
void hypertea_gpu_log(
  const int n, 
  const cl_mem a, 
  cl_mem y) {
  
  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "log_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&y));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&n));  

  size_t global_size = HYPERTEA_GET_BLOCKS(n);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  
  
  
}

template void hypertea_gpu_log<float>(
  const int n, 
  const cl_mem a, 
  cl_mem y
);

template void hypertea_gpu_log<half>(
  const int n, 
  const cl_mem a, 
  cl_mem y
);




template <typename Dtype>
void hypertea_gpu_powx(
  const int n, 
  const cl_mem a, 
  const float b, 
  cl_mem y) {
  
  Dtype b_(to_dtype_<Dtype>(b));

  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    "powx_kernel",
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&a),
      std::make_pair(sizeof(cl_mem), (void *)&y),
      std::make_pair(dtype_size_<Dtype>(), (void *)&b_),
      std::make_pair(sizeof(cl_int), (void *)&n)
    },
    std::vector<size_t> {HYPERTEA_GET_BLOCKS(n)},
    std::vector<size_t> {HYPERTEA_OPENCL_NUM_THREADS}
  );

}

template void hypertea_gpu_powx<float>(
  const int n, 
  const cl_mem a, 
  const float b, 
  cl_mem y
);

template void hypertea_gpu_powx<half>(
  const int n, 
  const cl_mem a, 
  const float b, 
  cl_mem y
);


template <typename Dtype>
void hypertea_gpu_sqrt(
  const int n, 
  const cl_mem a, 
  cl_mem y) {

  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    "sqrt_kernel",
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&a),
      std::make_pair(sizeof(cl_mem), (void *)&y),
      std::make_pair(sizeof(cl_int), (void *)&n)
    },
    std::vector<size_t> {HYPERTEA_GET_BLOCKS(n)},
    std::vector<size_t> {HYPERTEA_OPENCL_NUM_THREADS}
  );
  
}

template void hypertea_gpu_sqrt<float>(
  const int n, 
  const cl_mem a, 
  cl_mem y
);

template void hypertea_gpu_sqrt<half>(
  const int n, 
  const cl_mem a, 
  cl_mem y
);


template <typename Dtype>
void hypertea_gpu_dot(
  const int n,
  const cl_mem x,
  const cl_mem y,
  cl_mem out,
  int x_offset,
  int y_offset) {

    cl_mem temp_buffer = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);

    CLBLAST_CPP_CHECK(clblast::Dot<Dtype>(
         n,
         (cl_mem) temp_buffer, 0,
         (cl_mem) x, x_offset, 1,
         (cl_mem) y, y_offset, 1,
         &OpenCLHandler::Get().commandQueue, 
         NULL)
    );

    OPENCL_CHECK(clEnqueueReadBuffer(OpenCLHandler::Get().commandQueue, temp_buffer, CL_TRUE, 0, sizeof(float), out, 0, NULL, NULL));
    OPENCL_CHECK(clReleaseMemObject(temp_buffer));
}


template void hypertea_gpu_dot<float>(
  const int n,
  const cl_mem x,
  const cl_mem y,
  cl_mem out,
  int x_offset,
  int y_offset
);

template void hypertea_gpu_dot<half>(
  const int n,
  const cl_mem x,
  const cl_mem y,
  cl_mem out,
  int x_offset,
  int y_offset
);


template <typename Dtype>
void hypertea_gpu_asum(
  const int n, 
  const cl_mem x, 
  cl_mem y, 
  int x_offset) {

  cl_mem temp_buffer = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);

  CLBLAST_CPP_CHECK(clblast::Asum<Dtype>(
        n,
        (cl_mem) temp_buffer, 0,
        (cl_mem) x, x_offset, 1,
        &OpenCLHandler::Get().commandQueue, NULL));
  
  OPENCL_CHECK(clEnqueueReadBuffer(OpenCLHandler::Get().commandQueue, temp_buffer, CL_TRUE, 0, sizeof(float), y, 0, NULL, NULL));
  OPENCL_CHECK(clReleaseMemObject(temp_buffer));
}

template void hypertea_gpu_asum<float> (
  const int n, 
  const cl_mem x, 
  cl_mem y, 
  int x_offset
);

template void hypertea_gpu_asum<half> (
  const int n, 
  const cl_mem x, 
  cl_mem y, 
  int x_offset
);


template <typename Dtype>
void hypertea_gpu_fabs(
  const int n, 
  const cl_mem x, 
  cl_mem y) {
  
  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "abs_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&x));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&y));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&n));  

  size_t global_size = HYPERTEA_GET_BLOCKS(n);
    
}

template void hypertea_gpu_fabs<float>(
  const int n, 
  const cl_mem x, 
  cl_mem y
);

template void hypertea_gpu_fabs<half>(
  const int n, 
  const cl_mem x, 
  cl_mem y
);
 


template <typename Dtype>
void hypertea_gpu_scale(
  const int n,
  const float alpha,
  const cl_mem x,
  cl_mem y) {
  hypertea_cl_copy<Dtype>(n, x, y);
  hypertea_gpu_scal<Dtype>(n, alpha, y);
}

template void hypertea_gpu_scale<float>(
  const int n,
  const float alpha,
  const cl_mem x,
  cl_mem y
);

template void hypertea_gpu_scale<half>(
  const int n,
  const float alpha,
  const cl_mem x,
  cl_mem y
);



template <typename Dtype>
void hypertea_cl_copy(const int N, const cl_mem X, cl_mem Y, int x_offset, int y_offset) {
  if ((X != Y) || (x_offset != y_offset)) {
    OPENCL_CHECK(clEnqueueCopyBuffer(OpenCLHandler::Get().commandQueue, (cl_mem) X, (cl_mem) Y, 
                            x_offset, y_offset, sizeof(Dtype) * N, 0, NULL, NULL));
  }
}


template void hypertea_cl_copy<int>(const int N, const cl_mem X, cl_mem Y, int x_offset, int y_offset);
template void hypertea_cl_copy<unsigned int>(const int N, const cl_mem X,
    cl_mem Y, int x_offset, int y_offset);
template void hypertea_cl_copy<half>(const int N, const cl_mem X, cl_mem Y, int x_offset, int y_offset);
template void hypertea_cl_copy<float>(const int N, const cl_mem X, cl_mem Y, int x_offset, int y_offset);


}  // namespace hypertea

#endif //USE_OPENCL
