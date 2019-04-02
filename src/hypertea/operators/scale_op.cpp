#include <algorithm>
#include <vector>

#include "hypertea/operators/scale_op.hpp"

namespace hypertea {

template <>
TensorCPU<float> ScaleOp_CPU<float>::Forward(TensorCPU<float> &input_tensor) {


  const float* input_data = input_tensor.immutable_data();
  float* output_data = inplace_? input_tensor.mutable_data() : new float[input_tensor.size()];

  float* output_data_ptr_keeper = output_data;


  for (int n = 0; n < outer_dim_; ++n) {
    for (int d = 0; d < scale_dim_; ++d) {
      const float factor = scale_data_[d];
      hypertea_cpu_scale(inner_dim_, factor, input_data, output_data);
      input_data += inner_dim_;
      output_data += inner_dim_;
    } 
  }

  if (bias_data_ != NULL) {

    output_data = output_data_ptr_keeper;

    for (int n = 0; n < outer_dim_; ++n) {

      hypertea_cpu_gemm(CblasNoTrans, CblasNoTrans, scale_dim_,
          inner_dim_, 1, float(1), bias_data_,
          bias_multiplier_, float(1), output_data);
      output_data += (scale_dim_ * inner_dim_); 
    }
  }

  return inplace_? input_tensor:TensorCPU<float>(output_data_ptr_keeper, input_tensor.size());  

}

#ifdef USE_OPENCL


template <typename Dtype>
TensorGPU<Dtype> ScaleOp_GPU<Dtype>::Forward(TensorGPU<Dtype> input_tensor){

  TensorGPU<Dtype> output_tensor = inplace_? input_tensor : TensorGPU<Dtype>(input_tensor.count());

  if(!inplace_) {
    output_tensor.copy_data(input_tensor);
  }

  if (has_bias_) {
      inplace_channeled_scaladd(output_tensor, tweight_, tbias_, scale_dim_, inner_dim_);
  } else {
      inplace_channeled_scal(output_tensor, tweight_, scale_dim_, inner_dim_);
  }

  return output_tensor;


  // int data_count = input_tensor.count();

  // if (bias_data_) {

  //   cl_int ret;

  //   cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "ScaleBiasForward", &ret);
  //   OPENCL_CHECK(ret);


  //   // Set arguments for kernel
  //   OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_data));  
  //   OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output_data));  
  //   OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&data_count));  
  //   OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&scale_data_)); 
  //   OPENCL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&bias_data_));   
  //   OPENCL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&scale_dim_));  
  //   OPENCL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&inner_dim_));  

  //   size_t global_size = HYPERTEA_GET_BLOCKS(data_count);
    
  //   OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  


  // } else {


  //   cl_int ret;

  //   cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().math_program, "ScaleForward", &ret);
  //   OPENCL_CHECK(ret);

  //   // Set arguments for kernel
  //   OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_data));  
  //   OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output_data));  
  //   OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&data_count));  
  //   OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&scale_data_)); 
  //   OPENCL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&scale_dim_));  
  //   OPENCL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&inner_dim_));  

  //   size_t global_size = HYPERTEA_GET_BLOCKS(data_count);
    
  //   OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, &global_size, &HYPERTEA_OPENCL_NUM_THREADS, 0, NULL, NULL));  

  // }


  // return output_tensor;

}

#endif //USE_OPENCL

INSTANTIATE_CLASS_CPU(ScaleOp_CPU);
INSTANTIATE_CLASS_GPU(ScaleOp_GPU);
// REGISTER_LAYER_CLASS(Scale);

}  // namespace hypertea
