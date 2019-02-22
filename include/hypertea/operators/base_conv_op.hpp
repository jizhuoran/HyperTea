#ifndef HYPERTEA_BASE_CONVOLUTION_LAYER_HPP_
#define HYPERTEA_BASE_CONVOLUTION_LAYER_HPP_

#include <vector>
#include <iomanip>

#include "hypertea/util/im2col.hpp"
#include "hypertea/operator.hpp"

namespace hypertea {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
template <typename Dtype>
class BaseConvolutionOp_CPU : public CPUFunctor<Dtype> {
 public:

    explicit BaseConvolutionOp_CPU(const Dtype* weight, const Dtype* bias,
              int bottom_dim, int bottom_size, 
              int top_dim,

              int num,
              int channels,
              int group,
              int weight_offset,
              int num_output,
              int out_spatial_dim,
              bool is_1x1,
              bool force_nd_im2col,

              int conv_out_channels,
              int conv_in_channels,
              int conv_out_spatial_dim,
              int kernel_dim,
              int col_offset,
              int output_offset,

               int num_spatial_axes,
               std::vector<int> kernel_shape,
               std::vector<int> stride,
               std::vector<int> pad,
               std::vector<int> dilation,
               std::vector<int> conv_input_shape,
               std::vector<int> col_buffer_shape) 

    : CPUFunctor<Dtype>(),
      weight_(weight), bias_(bias),
      bottom_dim_(bottom_dim), bottom_size_(bottom_size),
      top_dim_(top_dim),
      
      num_(num),
      channels_(channels),
      group_(group),
      weight_offset_(weight_offset),
      num_output_(num_output),
      out_spatial_dim_(out_spatial_dim),
      is_1x1_(is_1x1),
      force_nd_im2col_(force_nd_im2col),

      conv_out_channels_(conv_out_channels),
      conv_in_channels_(conv_in_channels),
      conv_out_spatial_dim_(conv_out_spatial_dim),
      kernel_dim_(kernel_dim),
      col_offset_(col_offset),
      output_offset_(output_offset),

      num_spatial_axes_(num_spatial_axes),
      kernel_shape_(kernel_shape),
      stride_(stride),
      pad_(pad),
      dilation_(dilation),
      conv_input_shape_(conv_input_shape),
      col_buffer_shape_(col_buffer_shape) {

          bias_multiplier_ = (Dtype*)malloc(sizeof(Dtype) * out_spatial_dim_);
          hypertea_set(out_spatial_dim_, Dtype(1), bias_multiplier_);

          int col_buffer_size = 1;
          for (auto& n : col_buffer_shape)
            col_buffer_size *= n;

          col_buffer_ = (Dtype*)malloc(sizeof(Dtype) * col_buffer_size);
          hypertea_set(col_buffer_size, Dtype(1), col_buffer_);
      }






  const Dtype* weight_;
  const Dtype* bias_;
  int bottom_size_, bottom_dim_, top_dim_;


  int num_;
  int channels_;
  int group_;
  int weight_offset_;
  int num_output_;
  int out_spatial_dim_;
  bool is_1x1_;
  bool force_nd_im2col_;



  int num_spatial_axes_;
  std::vector<int> kernel_shape_;
  std::vector<int> stride_;
  std::vector<int> pad_;
  std::vector<int> dilation_;
  std::vector<int> conv_input_shape_;
  std::vector<int> col_buffer_shape_;
  // vector<int> output_shape_;

explicit BaseConvolutionOp_CPU(const Dtype* weight, const Dtype* bias,
              int bottom_dim, int bottom_size,
              int num,
              int input_channels,
              int group,
              int num_output,
              int out_spatial_dim,
              bool is_1x1,
              bool force_nd_im2col,
              int conv_out_spatial_dim,
              int num_spatial_axes,
              std::vector<int> kernel_shape,
              std::vector<int> stride,
              std::vector<int> pad,
              std::vector<int> dilation,
              std::vector<int> conv_input_shape,
              std::vector<int> col_buffer_shape,

              bool is_transposed
               ) 

    : CPUFunctor<Dtype>(),
      weight_(weight), bias_(bias),
      bottom_dim_(bottom_dim), bottom_size_(bottom_size),
      num_(num),
      group_(group),
      num_output_(num_output),
      out_spatial_dim_(out_spatial_dim),
      is_1x1_(is_1x1),
      force_nd_im2col_(force_nd_im2col),
      conv_out_spatial_dim_(conv_out_spatial_dim),
      num_spatial_axes_(num_spatial_axes),
      kernel_shape_(kernel_shape),
      stride_(stride),
      pad_(pad),
      dilation_(dilation),
      conv_input_shape_(conv_input_shape),
      col_buffer_shape_(col_buffer_shape) {





          if (is_transposed) {
            conv_out_channels_ = input_channels;
            conv_in_channels_ = num_output;
          } else {
            conv_out_channels_ = num_output;
            conv_in_channels_ = input_channels;
          }


          int filter_size = std::accumulate(kernel_shape.begin(), kernel_shape.end(), 1, std::multiplies<int>());

          kernel_dim_ = conv_in_channels_ * filter_size;
          weight_offset_ = conv_out_channels_ * kernel_dim_;

          col_offset_ = std::accumulate(col_buffer_shape_.begin(), col_buffer_shape_.end(), 1, std::multiplies<int>());
          output_offset_ = out_spatial_dim * conv_out_channels_;
          top_dim_ = out_spatial_dim * num_output;//output_offset_;

          // num_output_ = num_output;


          std::cout << bottom_dim_ << ", "
          << bottom_size_ << ", "
          << top_dim_ << ", "

          << num_ << ", "
          << input_channels << ", "
          << group_ << ", "
          << weight_offset_ << ", "
          << num_output << ", "
          << out_spatial_dim_ << ", "
          << false << ", "
          << false << ", "

          << conv_out_channels_ << ", "
          << conv_in_channels_ << ", "
          << conv_out_spatial_dim_ << ", "
          << kernel_dim_ << ", "
          << col_offset_ << ", "
          << output_offset_ << ", "

          << num_spatial_axes_ << std::endl;

          // std::cout << output_offset_ << std::endl;


          bias_multiplier_ = (Dtype*)malloc(sizeof(Dtype) * out_spatial_dim_);
          hypertea_set(out_spatial_dim_, Dtype(1), bias_multiplier_);

          int col_buffer_size = 1;
          for (auto& n : col_buffer_shape)
            col_buffer_size *= n;

          col_buffer_ = (Dtype*)malloc(sizeof(Dtype) * col_buffer_size);
          hypertea_set(col_buffer_size, Dtype(1), col_buffer_);
      }



protected:
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output);

private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_cpu(data, conv_in_channels_,
          conv_input_shape_[1], conv_input_shape_[2],
          kernel_shape_[0], kernel_shape_[1],
          pad_[0], pad_[1],
          stride_[0], stride_[1],
          dilation_[0], dilation_[1], col_buff);
    } else {
      im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.data(),
          col_buffer_shape_.data(), kernel_shape_.data(),
          pad_.data(), stride_.data(), dilation_.data(), col_buff);
    }
  }



  inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_cpu(col_buff, conv_in_channels_,
          conv_input_shape_[1], conv_input_shape_[2],
          kernel_shape_[0], kernel_shape_[1],
          pad_[0], pad_[1],
          stride_[0], stride_[1],
          dilation_[0], dilation_[1], data);
    } else {
      col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.data(),
          col_buffer_shape_.data(), kernel_shape_.data(),
          pad_.data(), stride_.data(), dilation_.data(), data);
    }
  }


//CPU NEEDED DATA
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int col_offset_;
  int output_offset_;

  Dtype* col_buffer_;
  Dtype* bias_multiplier_;
//CPU NEEDED DATA END


};

#ifdef USE_OPENCL

template <typename Dtype>
class BaseConvolutionOp_GPU : public GPUFunctor<Dtype> {

public:
  explicit BaseConvolutionOp_GPU(std::string kernel_name, int bottom_size,
                             cl_mem weight, cl_mem bias,
                             std::vector<int> local,
                             std::vector<int> global)

      : GPUFunctor<Dtype>(), kernel_name_(kernel_name), bottom_size_(bottom_size),
        weight_(weight), bias_(bias) {

          local_size_ = new size_t[3];
          local_size_[0] = local[0];
          local_size_[1] = local[1];
          local_size_[2] = local[2];

          global_size_ = new size_t[3];
          global_size_[0] = global[0];
          global_size_[1] = global[1];
          global_size_[2] = global[2];

        }


protected:

  const cl_mem weight_;
  const cl_mem bias_;
  int bottom_size_, bottom_dim_, top_dim_;


  std::string kernel_name_;
  size_t* local_size_;
  size_t* global_size_;

};

#endif //USE_OPENCL

}  // namespace hypertea

#endif  // HYPERTEA_BASE_CONVOLUTION_LAYER_HPP_
