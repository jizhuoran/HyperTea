#include <algorithm>
#include <vector>

#include "hypertea/operators/scale_op.hpp"

namespace hypertea {

template <>
void ScaleOp_CPU<float>::Forward(const std::vector<float*> bottom_datas,
      const std::vector<float*> top_datas) {

  float* tmp_top_data = top_datas[0];
  float* bottom_data = bottom_datas[0];

  for (int n = 0; n < outer_dim_; ++n) {
    for (int d = 0; d < scale_dim_; ++d) {
      const float factor = scale_data_[d];
      hypertea_cpu_scale(inner_dim_, factor, bottom_data, tmp_top_data);
      bottom_data += inner_dim_;
      tmp_top_data += inner_dim_;
    } 
  }


  tmp_top_data = top_datas[0];
  
  if (bias_data_) {

    float* bias_multiplier_ = (float* )malloc(sizeof(float) * inner_dim_);
    hypertea_set(inner_dim_, float(1), bias_multiplier_);

    for (int n = 0; n < outer_dim_; ++n) {

      hypertea_cpu_gemm(CblasNoTrans, CblasNoTrans, scale_dim_,
          inner_dim_, 1, float(1), bias_data_,
          bias_multiplier_, float(1), tmp_top_data);
      tmp_top_data += (scale_dim_ * inner_dim_);
    }
  }

}


template <>
std::vector<Tensor<float> *> ScaleOp_CPU<float>::Forward(std::vector<Tensor<float> *> inputs) {

  float* input = inputs[0]->data();
  Tensor<float>* output_tensor = new Tensor<float>(inputs[0]->size());
  float* output = output_tensor->data();


  for (int n = 0; n < outer_dim_; ++n) {
    for (int d = 0; d < scale_dim_; ++d) {
      const float factor = scale_data_[d];
      hypertea_cpu_scale(inner_dim_, factor, input, output);
      input += inner_dim_;
      output += inner_dim_;
    } 
  }


  output = output_tensor->data();
  
  if (bias_data_) {

    float* bias_multiplier_ = (float* )malloc(sizeof(float) * inner_dim_);
    hypertea_set(inner_dim_, float(1), bias_multiplier_);

    for (int n = 0; n < outer_dim_; ++n) {

      hypertea_cpu_gemm(CblasNoTrans, CblasNoTrans, scale_dim_,
          inner_dim_, 1, float(1), bias_data_,
          bias_multiplier_, float(1), output);
      output += (scale_dim_ * inner_dim_); 
    }
  }

  return {output_tensor};

}

#ifdef USE_OPENCL


template <typename Dtype>
void ScaleOp_GPU<Dtype>::Forward(const std::vector<cl_mem> bottom_datas,
      const std::vector<cl_mem> top_datas) {


  if (bias_data_) {

    cl_int ret;

    cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "ScaleBiasForward", &ret);
    OPENCL_CHECK(ret);

    // Set arguments for kernel
    OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bottom_datas[0]));  
    OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&top_datas[0]));  
    OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&top_count_));  
    OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&scale_data_)); 
    OPENCL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&bias_data_));   
    OPENCL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&scale_dim_));  
    OPENCL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&inner_dim_));  

    size_t global_size = HYPERTEA_GET_BLOCKS(top_count_);
    
    OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  


  } else {


    cl_int ret;

    cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "ScaleForward", &ret);
    OPENCL_CHECK(ret);

    // Set arguments for kernel
    OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bottom_datas[0]));  
    OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&top_datas[0]));  
    OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&top_count_));  
    OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&scale_data_)); 
    OPENCL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&scale_dim_));  
    OPENCL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&inner_dim_));  

    size_t global_size = HYPERTEA_GET_BLOCKS(top_count_);
    
    OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  


  }
}

#endif //USE_OPENCL

INSTANTIATE_CLASS_CPU(ScaleOp_CPU);
INSTANTIATE_CLASS_GPU(ScaleOp_GPU);
// REGISTER_LAYER_CLASS(Scale);

}  // namespace hypertea
