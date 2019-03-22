#include <vector>

#include "hypertea/operators/native_deconv_op.hpp"

namespace hypertea {




#ifdef USE_OPENCL




template <typename Dtype>
TensorGPU<Dtype> NativeDeconvolutionOp_GPU<Dtype>::Forward(TensorGPU<Dtype> input_tensor){


  const cl_mem input_data = input_tensor.immutable_data();
  TensorGPU<Dtype> output_tensor(top_count_);
  cl_mem output_data = output_tensor.mutable_data();
  cl_mem col_buff = col_buffer_;

  cl_int ret = -1;


  for (int i = 0; i < num_; ++i) {

    if (is_1x1_) {
      col_buff = output_data;
    }

    hypertea_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
          conv_out_spatial_dim_, conv_out_channels_,
          (float)1., weight_, input_data, (float)0., col_buff);


    int im_offset = i * top_size_;

    if (!is_1x1_) {
      
      cl_int ret = -1;

      cl_kernel col2im_kernel = clCreateKernel(OpenCLHandler::Get().conv_program, kernel_name_.c_str(), &ret);
      OPENCL_CHECK(ret);

      OPENCL_CHECK(clSetKernelArg(col2im_kernel, 0, sizeof(cl_mem), (void *)&col_buff));  
      OPENCL_CHECK(clSetKernelArg(col2im_kernel, 1, sizeof(cl_mem), (void *)&output_data));  
      OPENCL_CHECK(clSetKernelArg(col2im_kernel, 2, sizeof(cl_int), (void *)&im_offset));

      OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, col2im_kernel, 3, NULL, global_size_, local_size_, 0, NULL, NULL));  

    }


    if (bias_) {

      hypertea_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
        out_spatial_dim_, 1, (Dtype)1., bias_, bias_multiplier_,
        (Dtype)1., output_data);
    }


  }


  return output_tensor;

}


#endif //USE_OPENCL




INSTANTIATE_CLASS_GPU(NativeDeconvolutionOp_GPU);
// REGISTER_LAYER_CLASS(Deconvolution);

}  // namespace hypertea
