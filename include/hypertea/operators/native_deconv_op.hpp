#ifndef HYPERTEA_NATIVE_DECONV_LAYER_HPP_
#define HYPERTEA_NATIVE_DECONV_LAYER_HPP_

#include <vector>

#include "hypertea/operator.hpp"
#include "hypertea/util/math_functions.hpp"

namespace hypertea {


#ifdef USE_OPENCL

template <typename Dtype>
class NativeDeconvolutionOp_GPU : public GPUFunctor<Dtype>{
 public:

  explicit NativeDeconvolutionOp_GPU(std::string kernel_name, int top_size, int kernel_dim,
                             cl_mem weight, cl_mem bias,
                             std::vector<int> input_shape,
                             std::vector<int> output_shape,
                             bool is_1x1,
                             std::vector<int> local,
                             std::vector<int> global)

      : GPUFunctor<Dtype>(), kernel_name_(kernel_name), 
        top_size_(top_size), kernel_dim_(kernel_dim),
        weight_(weight), bias_(bias),
        is_1x1_(is_1x1) { 

        local_size_ = new size_t[3];
        local_size_[0] = local[0];
        local_size_[1] = local[1];
        local_size_[2] = local[2];

        global_size_ = new size_t[3];
        global_size_[0] = global[0];
        global_size_[1] = global[1];
        global_size_[2] = global[2];

        num_ = input_shape[0];
        num_output_ = output_shape[1];
        top_count_ = num_ * top_size;
        out_spatial_dim_ = std::accumulate(output_shape.begin()+2, output_shape.end(), 1, std::multiplies<int>());



        conv_out_channels_ = input_shape[1];
        conv_out_spatial_dim_ = std::accumulate(input_shape.begin()+2, input_shape.end(), 1, std::multiplies<int>());


        col_offset_ = kernel_dim_ * conv_out_spatial_dim_;

        bias_multiplier_ = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, out_spatial_dim_ * sizeof(Dtype), NULL, NULL);
        hypertea_gpu_set<Dtype>(out_spatial_dim_, Dtype(1), bias_multiplier_);
        col_buffer_ = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, col_offset_ * sizeof(Dtype), NULL, NULL);
        hypertea_gpu_set<Dtype>(col_offset_, Dtype(1), col_buffer_);


      }

  virtual inline const char* type() const { return "Deconvolution"; }

 // protected:
  // virtual void Forward(const std::vector<cl_mem> bottom_datas,
  //     const std::vector<cl_mem> top_datas);

  virtual TensorGPU<Dtype> Forward(TensorGPU<Dtype> input_tensor);


private:

  std::string kernel_name_;
  int top_size_;

  cl_mem weight_ = NULL;
  cl_mem bias_ = NULL;
  cl_mem bias_multiplier_;
  cl_mem col_buffer_;

  int top_count_;

  size_t* local_size_;
  size_t* global_size_;

  int num_output_;
  int out_spatial_dim_;


  int num_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int conv_out_channels_;
  int col_offset_;
  bool is_1x1_;
  
};

#endif //USE_OPENCL


}  // namespace hypertea

#endif  // HYPERTEA_NATIVE_DECONV_LAYER_HPP_
