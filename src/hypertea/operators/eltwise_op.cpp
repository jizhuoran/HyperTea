#include <cfloat>
#include <vector>

#include "hypertea/operators/eltwise_op.hpp"
#include "hypertea/util/math_functions.hpp"

namespace hypertea {



template <>
void EltwiseOp_CPU<float>::Forward(const std::vector<float*> bottom_datas,
      const std::vector<float*> top_datas) {

  float* top_data = top_datas[0];

  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    hypertea_mul(top_count_, bottom_datas[0], bottom_datas[1], top_data);
    for (int i = 2; i < bottom_nums_; ++i) {
      hypertea_mul(top_count_, top_data, bottom_datas[i], top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    hypertea_set<float>(top_count_, float(0), top_data);
    // TODO(shelhamer) does BLAS optimize to sum for coeff = 1?
    for (int i = 0; i < bottom_nums_; ++i) {
      hypertea_axpy(top_count_, coeffs_[i], bottom_datas[i], top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_MAX:
    // Initialize

    hypertea_set(top_count_, -1, max_idx_);
    hypertea_set<float>(top_count_, float(-FLT_MAX), top_data);
    // bottom 0 & 1
    for (int idx = 0; idx < top_count_; ++idx) {
      if (bottom_datas[0][idx] > bottom_datas[1][idx]) {
        top_data[idx] = bottom_datas[0][idx];  // maxval
        max_idx_[idx] = 0;  // maxid
      } else {
        top_data[idx] = bottom_datas[1][idx];  // maxval
        max_idx_[idx] = 1;  // maxid
      }
    }
    // bottom 2++
    for (int blob_idx = 2; blob_idx < bottom_nums_; ++blob_idx) {
      for (int idx = 0; idx < top_count_; ++idx) {
        if (bottom_datas[blob_idx][idx] > top_data[idx]) {
          top_data[idx] = bottom_datas[blob_idx][idx];  // maxval
          max_idx_[idx] = blob_idx;  // maxid
        }
      }
    }
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}


#ifdef USE_OPENCL

template <typename Dtype>
void EltwiseOp_GPU<Dtype>::Forward(const std::vector<cl_mem> bottom_datas,
      const std::vector<cl_mem> top_datas) {

  int blob_idx = 0;
  size_t global_size = 0;
 
  cl_int ret;
  cl_kernel kernel;

  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    hypertea_gpu_mul<Dtype>(top_count_, bottom_datas[0], bottom_datas[1],
        top_datas[0]);
    for (int i = 2; i < bottom_nums_; ++i) {
      hypertea_gpu_mul<Dtype>(top_count_, top_datas[0], bottom_datas[i], top_datas[0]);
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    hypertea_gpu_set<Dtype>(top_count_, Dtype(0.), top_datas[0]);
    // TODO(shelhamer) does cuBLAS optimize to sum for coeff = 1?
    for (int i = 0; i < bottom_nums_; ++i) {
      hypertea_gpu_axpy<Dtype>(top_count_, coeffs_[i], bottom_datas[i], top_datas[0]);
    }
    break;
  case EltwiseParameter_EltwiseOp_MAX:
    // mask = max_idx_.mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)

    blob_idx = 0;

    kernel = clCreateKernel(OpenCLHandler::Get().math_program, "MaxForward", &ret);
    OPENCL_CHECK(ret);

    // Set arguments for kernel
    OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bottom_datas[0]));  
    OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bottom_datas[1]));  
    OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&top_datas[0]));  
    OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&max_idx_));  
    OPENCL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&top_count_));  
    OPENCL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&blob_idx));  

    global_size = HYPERTEA_GET_BLOCKS(top_count_);
    
    OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  

    for (int i = 2; i < bottom_nums_; ++i) {
      // NOLINT_NEXT_LINE(whitespace/operators)

      blob_idx = i-1;

      OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&top_datas[0]));  
      OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bottom_datas[i]));  
      OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&top_datas[0]));  
      OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&max_idx_));  
      OPENCL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&top_count_));  
      OPENCL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&blob_idx));  

    
      OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  


    }
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }

}

#endif //USE_OPENCL


INSTANTIATE_CLASS_CPU(EltwiseOp_CPU);
INSTANTIATE_CLASS_GPU(EltwiseOp_GPU);
// REGISTER_LAYER_CLASS(Eltwise);

}  // namespace hypertea
