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

  CLBLAST_CPP_CHECK(clblast::Gemm<Dtype>(clblast::Layout::kRowMajor,
                                          blastTransA, blastTransB,
                                          M, N, K,
                                          alpha_,
                                          (cl_mem) A, 0, lda,
                                          (cl_mem) B, 0, ldb,
                                          beta_,
                                          (cl_mem) C, 0, ldc,
                                          &OpenCLHandler::Get().commandQueue, NULL));
}

template void hypertea_gpu_gemm<float>(
  const CBLAS_TRANSPOSE TransA,
  const CBLAS_TRANSPOSE TransB, 
  const int M, const int N, const int K,
  const float alpha, 
  const cl_mem A, 
  const cl_mem B, 
  const float beta,
  cl_mem C);

template void hypertea_gpu_gemm<half>(
  const CBLAS_TRANSPOSE TransA,
  const CBLAS_TRANSPOSE TransB, 
  const int M, const int N, const int K,
  const float alpha, 
  const cl_mem A, 
  const cl_mem B, 
  const float beta,
  cl_mem C);

// template <>
// void hypertea_gpu_gemm<half>(const CBLAS_TRANSPOSE TransA,
//     const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
//     const float alpha, const cl_mem A, const cl_mem B, const float beta,
//     cl_mem C) {
  
//   size_t lda = (TransA == CblasNoTrans) ? K : M;
//   size_t ldb = (TransB == CblasNoTrans) ? N : K;
//   size_t ldc = N;

//   CLBlastTranspose_ blastTransA =
//       (TransA == CblasNoTrans) ? CLBlastTransposeNo : CLBlastTransposeYes;
//   CLBlastTranspose_ blastTransB =
//       (TransB == CblasNoTrans) ? CLBlastTransposeNo : CLBlastTransposeYes;


//   half alpha_half = float2half_impl(alpha);
//   half beta_half = float2half_impl(beta);

//   CLBLAST_CHECK(CLBlastHgemm(CLBlastLayoutRowMajor,
//                               blastTransA, blastTransB,
//                               M, N, K,
//                               alpha_half,
//                               (cl_mem) A, 0, lda,
//                               (cl_mem) B, 0, ldb,
//                               beta_half,
//                               (cl_mem) C, 0, ldc,
//                               &OpenCLHandler::Get().commandQueue, NULL));

// }


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

  CLBLAST_CPP_CHECK(clblast::Gemv<Dtype>(clblast::Layout::kColMajor,
                                blastTransA, 
                                N, M,
                                alpha_,
                                (cl_mem) A, 0, N,
                                (cl_mem) x, 0, 1,
                                beta_,
                                (cl_mem) y, 0, 1,
                                &OpenCLHandler::Get().commandQueue, NULL));
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


// template <>
// void hypertea_gpu_gemv<half>(const CBLAS_TRANSPOSE TransA, const int M, const int N,
//     const float alpha, const cl_mem A, const cl_mem x, const float beta,
//     cl_mem y) {
//       CLBlastTranspose_ blastTransA =
//       (TransA != CblasNoTrans) ? CLBlastTransposeNo : CLBlastTransposeYes;



//     half alpha_half = float2half_impl(alpha);
//     half beta_half = float2half_impl(beta);

//     CLBLAST_CHECK(CLBlastHgemv(CLBlastLayoutColMajor, 
//                                             blastTransA, 
//                                             N, M,
//                                             (cl_half) alpha_half,
//                                             (cl_mem) A, 0, N,
//                                             (cl_mem) x, 0, 1,
//                                             (cl_half) beta_half,
//                                             (cl_mem) y, 0, 1,
//                                             &OpenCLHandler::Get().commandQueue, NULL));

// }

template <>
void hypertea_gpu_bsum<float>(const int m, const int n, const cl_mem X, const float alpha, const float beta,
                            cl_mem y, const int x_inc) {

  cl_int ret;

  cl_kernel kernel1 = clCreateKernel(OpenCLHandler::Get().math_program, "Xasum", &ret);
  OPENCL_CHECK(ret);
  cl_kernel kernel2 = clCreateKernel(OpenCLHandler::Get().math_program, "XasumEpilogue", &ret);
  OPENCL_CHECK(ret);


  size_t temp_size = 2*64;

  cl_mem temp_buffer = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, m * temp_size * sizeof(float), NULL, NULL);

  OPENCL_CHECK(clSetKernelArg(kernel1, 0, sizeof(cl_int), (void *)&n));  
  OPENCL_CHECK(clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void *)&X));  
  OPENCL_CHECK(clSetKernelArg(kernel1, 2, sizeof(cl_int), (void *)&x_inc));  
  OPENCL_CHECK(clSetKernelArg(kernel1, 3, sizeof(cl_mem), (void *)&temp_buffer));
  OPENCL_CHECK(clSetKernelArg(kernel1, 4, sizeof(cl_float), (void *)&alpha));  



  size_t* local_size = new size_t[2];
  local_size[0] = static_cast<size_t>(64);
  local_size[1] = static_cast<size_t>(1);

  size_t* global_size = new size_t[2];
  global_size[0] = static_cast<size_t>(temp_size * 64);
  global_size[1] = static_cast<size_t>(m);



  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel1, 2, NULL, global_size, local_size, 0, NULL, NULL));  


  OPENCL_CHECK(clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&temp_buffer));  
  OPENCL_CHECK(clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void *)&y));
  OPENCL_CHECK(clSetKernelArg(kernel2, 2, sizeof(cl_float), (void *)&beta));

  global_size[0] = static_cast<size_t>(64);

  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel2, 2, NULL, global_size, local_size, 0, NULL, NULL));  
 
  // OPENCL_CHECK(clReleaseMemObject(temp_buffer));


}


template <>
void hypertea_gpu_bsum<half>(const int m, const int n, const cl_mem X, const float alpha,  const float beta,
                            cl_mem y, const int x_inc) {
  
  cl_int ret;
  cl_kernel kernel1 = clCreateKernel(OpenCLHandler::Get().math_program, "Xasum", &ret);
  OPENCL_CHECK(ret);
  cl_kernel kernel2 = clCreateKernel(OpenCLHandler::Get().math_program, "XasumEpilogue", &ret);
  OPENCL_CHECK(ret);

  half alpha_half = float2half_impl(alpha);


  size_t temp_size = 2*64;

  cl_mem temp_buffer = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, m * temp_size * sizeof(half), NULL, NULL);

  OPENCL_CHECK(clSetKernelArg(kernel1, 0, sizeof(cl_int), (void *)&n));  
  OPENCL_CHECK(clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void *)&X));  
  OPENCL_CHECK(clSetKernelArg(kernel1, 2, sizeof(cl_int), (void *)&x_inc));  
  OPENCL_CHECK(clSetKernelArg(kernel1, 3, sizeof(cl_mem), (void *)&temp_buffer));
  OPENCL_CHECK(clSetKernelArg(kernel1, 4, sizeof(cl_half), (void *)&alpha_half));  


  size_t* local_size = new size_t[2];
  local_size[0] = static_cast<size_t>(64);
  local_size[1] = static_cast<size_t>(1);

  size_t* global_size = new size_t[2];
  global_size[0] = static_cast<size_t>(temp_size * 64);
  global_size[1] = static_cast<size_t>(m);



  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel1, 2, NULL, global_size, local_size, 0, NULL, NULL));  


  half beta_half = float2half_impl(beta);
  OPENCL_CHECK(clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&temp_buffer));  
  OPENCL_CHECK(clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void *)&y));
  OPENCL_CHECK(clSetKernelArg(kernel2, 2, sizeof(cl_half), (void *)&beta_half));

  global_size[0] = static_cast<size_t>(64);

  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel2, 2, NULL, global_size, local_size, 0, NULL, NULL));  
 

}


template <>
void hypertea_gpu_axpy<float>(const int N, const float alpha, const cl_mem X,
    cl_mem Y) {
      
  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "axpy_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&X));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&Y));
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_float), (void *)&alpha));
  OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&N));  

  size_t global_size = HYPERTEA_GET_BLOCKS(N);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  
  
}

template <>
void hypertea_gpu_axpy<half>(const int N, const float alpha, const cl_mem X,
    cl_mem Y) {
  
  cl_int ret;
  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "axpy_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  half alpha_half = float2half_impl(alpha);

  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&X));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&Y));
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_half), (void *)&alpha_half));
  OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&N));  

  size_t global_size = HYPERTEA_GET_BLOCKS(N);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  
  
}


template <>
void hypertea_gpu_set<float>(const int N, const float alpha, cl_mem Y) {
  OPENCL_CHECK(clEnqueueFillBuffer(OpenCLHandler::Get().commandQueue, (cl_mem) Y, &alpha, sizeof(float), 0, N * sizeof(float), 0, NULL, NULL));
}

template <>
void hypertea_gpu_set<half>(const int N, const float alpha, cl_mem Y) {

  half alpha_half = float2half_impl(alpha);
  OPENCL_CHECK(clEnqueueFillBuffer(OpenCLHandler::Get().commandQueue, (cl_mem) Y, &alpha_half, sizeof(half), 0, N * sizeof(half), 0, NULL, NULL));

}

void hypertea_gpu_set(const int N, const int alpha, cl_mem Y) {
  OPENCL_CHECK(clEnqueueFillBuffer(OpenCLHandler::Get().commandQueue, (cl_mem) Y, &alpha, sizeof(int), 0, N * sizeof(int), 0, NULL, NULL));
}


template <>
void hypertea_gpu_add_scalar<float>(const int N, const float alpha, cl_mem X) {
  
  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "add_scalar_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&X));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_float), (void *)&alpha));
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&N));  

  size_t global_size = HYPERTEA_GET_BLOCKS(N);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  
  
}


template <>
void hypertea_gpu_add_scalar<half>(const int N, const float alpha, cl_mem X) {
  
  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "add_scalar_kernel", &ret);
  OPENCL_CHECK(ret);

  half alpha_half = float2half_impl(alpha);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&X));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_half), (void *)&alpha_half));
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&N));  

  size_t global_size = HYPERTEA_GET_BLOCKS(N);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  
  
}




template <>
void hypertea_gpu_scal<float>(const int N, const float alpha, cl_mem X){
  
  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "scal_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&X));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_float), (void *)&alpha));
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&N));  

  size_t global_size = HYPERTEA_GET_BLOCKS(N);

  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  
 
}


template <>
void hypertea_gpu_scal<half>(const int N, const float alpha, cl_mem X){
  
  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "scal_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  half alpha_half = float2half_impl(alpha);

  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&X));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_half), (void *)&alpha_half));
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&N));  

  size_t global_size = HYPERTEA_GET_BLOCKS(N);

  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  
 
}



template <>
void hypertea_gpu_axpby<float>(const int N, const float alpha, const cl_mem X,
    const float beta, cl_mem Y) {
  hypertea_gpu_scal<float>(N, beta, Y);
  hypertea_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void hypertea_gpu_axpby<half>(const int N, const float alpha, const cl_mem X,
    const float beta, cl_mem Y) {
  hypertea_gpu_scal<half>(N, beta, Y);
  hypertea_gpu_axpy<half>(N, alpha, X, Y);
}

template <>
void hypertea_gpu_add<float>(const int N, const cl_mem a, const cl_mem b, cl_mem y){
    
  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "add_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&y));  
  OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&N));  

  size_t global_size = HYPERTEA_GET_BLOCKS(N);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  
  
}

template <>
void hypertea_gpu_add<half>(const int N, const cl_mem a, const cl_mem b, cl_mem y){
  
  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "add_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&y));  
  OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&N));  

  size_t global_size = HYPERTEA_GET_BLOCKS(N);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  
  
}


template <>
void hypertea_gpu_sub<float>(const int N, const cl_mem a, const cl_mem b, cl_mem y){
    
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

template <>
void hypertea_gpu_sub<half>(const int N, const cl_mem a, const cl_mem b, cl_mem y){
  
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

template <>
void hypertea_gpu_mul<float>(const int N, const cl_mem a, const cl_mem b, cl_mem y){
  
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



template <>
void hypertea_gpu_mul<half>(const int N, const cl_mem a, const cl_mem b, cl_mem y){
  
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
template <>
void hypertea_gpu_div<float>(const int N, const cl_mem a, const cl_mem b, cl_mem y){
    
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


template <>
void hypertea_gpu_div<half>(const int N, const cl_mem a, const cl_mem b, cl_mem y){
  
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


template <>
void hypertea_gpu_abs<float>(const int n, const cl_mem a, cl_mem y){
  
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


template <>
void hypertea_gpu_abs<half>(const int n, const cl_mem a, cl_mem y){
  
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


template <>
void hypertea_gpu_exp<float>(const int n, const cl_mem a, cl_mem y){
  
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


template <>
void hypertea_gpu_exp<half>(const int n, const cl_mem a, cl_mem y){
  
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




template <>
void hypertea_gpu_log<float>(const int n, const cl_mem a, cl_mem y){
  
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


template <>
void hypertea_gpu_log<half>(const int n, const cl_mem a, cl_mem y){
  
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



template <>
void hypertea_gpu_powx<float>(const int n, const cl_mem a, const float b, cl_mem y){
  
  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "powx_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&y));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_float), (void *)&b));  
  OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&n));  

  size_t global_size = HYPERTEA_GET_BLOCKS(n);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  
  
}


template <>
void hypertea_gpu_powx<half>(const int n, const cl_mem a, const float b, cl_mem y){
  
  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "powx_kernel", &ret);
  OPENCL_CHECK(ret);

  half b_half = float2half_impl(b);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&y));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_half), (void *)&b_half));  
  OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&n));  

  size_t global_size = HYPERTEA_GET_BLOCKS(n);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  
  
}




template <>
void hypertea_gpu_sqrt<float>(const int n, const cl_mem a, cl_mem y){
      
  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "sqrt_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&y));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&n));  

  size_t global_size = HYPERTEA_GET_BLOCKS(n);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  
  
}

// hypertea_gpu_rng_uniform with two arguments generates integers in the range
// [0, UINT_MAX].


template <>
void hypertea_gpu_rng_uniform<float>(const int n, const float a, const float b, cl_mem r){
  NOT_IMPLEMENT;
}

template <>
void hypertea_gpu_rng_gaussian<float>(const int n, const float mu, const float sigma,
                            cl_mem r){
  NOT_IMPLEMENT;
}

template <>
void hypertea_gpu_rng_bernoulli<float>(const int n, const float p, int* r){
  NOT_IMPLEMENT;
}

template <>
void hypertea_gpu_dot<float>(const int n, const cl_mem x, const cl_mem y, cl_mem out, int x_offset, int y_offset){

    cl_mem temp_buffer = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);

    CLBLAST_CHECK(CLBlastSdot(n,
         (cl_mem) temp_buffer, 0,
         (cl_mem) x, x_offset, 1,
         (cl_mem) y, y_offset, 1,
         &OpenCLHandler::Get().commandQueue, NULL));

    OPENCL_CHECK(clEnqueueReadBuffer(OpenCLHandler::Get().commandQueue, temp_buffer, CL_TRUE, 0, sizeof(float), out, 0, NULL, NULL));
    OPENCL_CHECK(clReleaseMemObject(temp_buffer));
}


template <>
void hypertea_gpu_dot<half>(const int n, const cl_mem x, const cl_mem y, cl_mem out, int x_offset, int y_offset){

  cl_mem temp_buffer = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, sizeof(half), NULL, NULL);


  CLBLAST_CHECK(CLBlastHdot(n,
         (cl_mem) temp_buffer, 0,
         (cl_mem) x, x_offset, 1,
         (cl_mem) y, y_offset, 1,
         &OpenCLHandler::Get().commandQueue, NULL));
  
  OPENCL_CHECK(clEnqueueReadBuffer(OpenCLHandler::Get().commandQueue, temp_buffer, CL_TRUE, 0, sizeof(half), out, 0, NULL, NULL));
  OPENCL_CHECK(clReleaseMemObject(temp_buffer));
}






template <>
void hypertea_gpu_asum<float>(const int n, const cl_mem x, cl_mem y, int x_offset){

  cl_mem temp_buffer = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);

  CLBLAST_CHECK(CLBlastSasum(n,
        (cl_mem) temp_buffer, 0,
        (cl_mem) x, x_offset, 1,
        &OpenCLHandler::Get().commandQueue, NULL));
  
  OPENCL_CHECK(clEnqueueReadBuffer(OpenCLHandler::Get().commandQueue, temp_buffer, CL_TRUE, 0, sizeof(float), y, 0, NULL, NULL));
  OPENCL_CHECK(clReleaseMemObject(temp_buffer));
}


template <>
void hypertea_gpu_asum<half>(const int n, const cl_mem x, cl_mem y, int x_offset){

  cl_mem temp_buffer = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, sizeof(half), NULL, NULL);

  CLBLAST_CHECK(CLBlastHasum(n,
        (cl_mem) temp_buffer, 0,
        (cl_mem) x, x_offset, 1,
        &OpenCLHandler::Get().commandQueue, NULL));

  OPENCL_CHECK(clEnqueueReadBuffer(OpenCLHandler::Get().commandQueue, temp_buffer, CL_TRUE, 0, sizeof(half), y, 0, NULL, NULL));
  OPENCL_CHECK(clReleaseMemObject(temp_buffer));
}



template<>
void hypertea_gpu_sign<float>(const int n, const cl_mem x, cl_mem y){
  NOT_IMPLEMENT;
}

template<>
void hypertea_gpu_sign<half>(const int n, const cl_mem x, cl_mem y){
  NOT_IMPLEMENT;
}


template<>
void hypertea_gpu_sgnbit<float>(const int n, const cl_mem x, cl_mem y){
  NOT_IMPLEMENT;
}

template<>
void hypertea_gpu_sgnbit<half>(const int n, const cl_mem x, cl_mem y){
  NOT_IMPLEMENT;
}


template <>
void hypertea_gpu_fabs<float>(const int n, const cl_mem x, cl_mem y){
  
  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "abs_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&x));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&y));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&n));  

  size_t global_size = HYPERTEA_GET_BLOCKS(n);
  
  
}

template <>
void hypertea_gpu_fabs<half>(const int n, const cl_mem x, cl_mem y){
  
  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "abs_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&x));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&y));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&n));  

  size_t global_size = HYPERTEA_GET_BLOCKS(n);
  
  
}

template <>
void hypertea_gpu_scale<float>(const int n, const float alpha, const cl_mem x, cl_mem y){
  hypertea_cl_copy<float>(n, x, y);
  hypertea_gpu_scal<float>(n, alpha, y);
}

template <>
void hypertea_gpu_scale<half>(const int n, const float alpha, const cl_mem x, cl_mem y){
  hypertea_cl_copy<half>(n, x, y);
  hypertea_gpu_scal<half>(n, alpha, y);
}



template <>
void hypertea_gpu_sqrt<half>(const int n, const cl_mem a, cl_mem y){
  
  cl_int ret;

  cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "sqrt_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&y));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&n));  

  size_t global_size = HYPERTEA_GET_BLOCKS(n);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  
  
}




void hypertea_gpu_memcpy(const size_t N, const void* X, void* Y) {

  if (X != Y) {
    cl_int err = clEnqueueCopyBuffer(OpenCLHandler::Get().commandQueue, (cl_mem) X, (cl_mem) Y, 0, 0, N, 0,  NULL, NULL);
  }

}


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
