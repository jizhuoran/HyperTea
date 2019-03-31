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
              int group, bool is_1x1,
              std::vector<int> kernel_shape,
              std::vector<int> stride,
              std::vector<int> pad,
              std::vector<int> dilation,
              std::vector<int> input_shape,
              std::vector<int> output_shape,
              bool force_nd_im2col,
              bool is_transposed) 
    : CPUFunctor<Dtype>(),
      weight_(weight), bias_(bias),
      group_(group),
      is_1x1_(is_1x1),
      kernel_shape_(kernel_shape),
      stride_(stride),
      pad_(pad),
      dilation_(dilation),
      force_nd_im2col_(force_nd_im2col) {


          num_ = input_shape[0];
          num_output_ = output_shape[1];
          num_spatial_axes_ = kernel_shape.size();

          bottom_dim_ = std::accumulate(std::next(input_shape.begin()), input_shape.end(), 1, std::multiplies<int>());
          out_spatial_dim_ = std::accumulate(output_shape.begin()+2, output_shape.end(), 1, std::multiplies<int>());
          top_dim_ = out_spatial_dim_ * num_output_;//output_offset_;
          
          top_count_ = top_dim_ * num_;


          col_buffer_shape_.push_back(group_);

          if (is_transposed) {
            conv_out_channels_ = input_shape[1];
            conv_in_channels_ = output_shape[1];
            conv_input_shape_ = std::vector<int> ((output_shape.begin() + num_spatial_axes_ - 1), output_shape.end());
            col_buffer_shape_.insert(col_buffer_shape_.end(), input_shape.begin()+2, input_shape.end());
            conv_out_spatial_dim_ = std::accumulate(input_shape.begin()+2, input_shape.end(), 1, std::multiplies<int>());
          } else {
            conv_out_channels_ = output_shape[1];
            conv_in_channels_ = input_shape[1];
            conv_input_shape_ = std::vector<int> ((input_shape.begin() + num_spatial_axes_ - 1), input_shape.end());
            col_buffer_shape_.insert(col_buffer_shape_.end(), output_shape.begin()+2, output_shape.end());
            conv_out_spatial_dim_ = std::accumulate(output_shape.begin()+2, output_shape.end(), 1, std::multiplies<int>());
          }

          kernel_dim_ = conv_in_channels_ * std::accumulate(kernel_shape.begin(), kernel_shape.end(), 1, std::multiplies<int>());
          weight_offset_ = conv_out_channels_ * kernel_dim_;
          col_buffer_shape_[0] *= kernel_dim_;

          col_offset_ = std::accumulate(col_buffer_shape_.begin(), col_buffer_shape_.end(), 1, std::multiplies<int>());
          output_offset_ = conv_out_spatial_dim_ * conv_out_channels_ / group_;

 
          // std::cout << bottom_dim_ << ", "
          // << 1 << ", "
          // << top_dim_ << ", "

          // << num_ << ", "
          // << input_shape[1] << ", "
          // << group_ << ", "
          // << weight_offset_ << ", "
          // << num_output_ << ", "
          // << out_spatial_dim_ << ", "
          // << false << ", "
          // << false << ", "

          // << conv_out_channels_ << ", "
          // << conv_in_channels_ << ", "
          // << conv_out_spatial_dim_ << ", "
          // << kernel_dim_ << ", "
          // << col_offset_ << ", "
          // << output_offset_ << ", "

          // << num_spatial_axes_ << std::endl;




          bias_multiplier_ = new Dtype[out_spatial_dim_];
          hypertea_set(out_spatial_dim_, Dtype(1), bias_multiplier_);
          col_buffer_ = new Dtype[col_offset_];
          hypertea_set(col_offset_, Dtype(1), col_buffer_);
      }


  ~BaseConvolutionOp_CPU() {
    delete [] bias_multiplier_;
    delete [] col_buffer_;
  }



  const Dtype* weight_;
  const Dtype* bias_;
  int bottom_dim_ = -1;
  int top_dim_ = -1;
  int num_ = -1;
  int group_ = -1;
  int weight_offset_ = -1;
  int num_output_ = -1;
  int out_spatial_dim_ = -1;
  int top_count_ = -1;
  bool is_1x1_;
  bool force_nd_im2col_;



  int num_spatial_axes_ = -1;
  std::vector<int> kernel_shape_;
  std::vector<int> stride_;
  std::vector<int> pad_;
  std::vector<int> dilation_;
  std::vector<int> conv_input_shape_;
  std::vector<int> col_buffer_shape_;





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
  explicit BaseConvolutionOp_GPU(std::string kernel_name, int top_count,
                             cl_mem weight, cl_mem bias,
                             std::vector<int> local,
                             std::vector<int> global)

      : GPUFunctor<Dtype>(), kernel_name_(kernel_name), top_count_(top_count),
        weight_(weight), bias_(bias) {

          local_size_.push_back(local[0]);
          local_size_.push_back(local[1]);
          local_size_.push_back(local[2]);

          global_size_.push_back(global[0]);
          global_size_.push_back(global[1]);
          global_size_.push_back(global[2]);

        }


protected:

  const cl_mem weight_;
  const cl_mem bias_;
  int top_count_;


  std::string kernel_name_;
  std::vector<size_t> local_size_;
  std::vector<size_t> global_size_;

};

#endif //USE_OPENCL

}  // namespace hypertea

#endif  // HYPERTEA_BASE_CONVOLUTION_LAYER_HPP_
