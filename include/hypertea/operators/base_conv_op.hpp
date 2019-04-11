#ifndef HYPERTEA_BASE_CONVOLUTION_OP_HPP_
#define HYPERTEA_BASE_CONVOLUTION_OP_HPP_

#include <vector>
#include <iomanip>

#include "hypertea/util/im2col.hpp"
#include "hypertea/operator.hpp"

namespace hypertea {


template <typename DeviceTensor>
class BaseConvolutionOp : public TensorOperator<DeviceTensor>{
 public:

    

  explicit BaseConvolutionOp(
    DeviceTensor* weight, 
    DeviceTensor* bias,
    int group, bool is_1x1,
    std::vector<int> kernel_shape,
    std::vector<int> stride,
    std::vector<int> pad,
    std::vector<int> dilation,
    std::vector<int> input_shape,
    std::vector<int> output_shape,
    bool is_transposed) 
    : TensorOperator<DeviceTensor>(),
      weight_(weight), 
      bias_(bias),
      group_(group),
      is_1x1_(is_1x1),
      kernel_shape_(kernel_shape),
      stride_(stride),
      pad_(pad),
      dilation_(dilation) {


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

          if(!is_1x1) {
            col_buffer_ = new DeviceTensor(col_offset_);
          }


          
      }


  ~BaseConvolutionOp() {
    if(!is_1x1_) {
      delete col_buffer_;
    }
  }


  DeviceTensor* weight_;
  DeviceTensor* bias_;

  int bottom_dim_ = -1;
  int top_dim_ = -1;
  int num_ = -1;
  int group_ = -1;
  int weight_offset_ = -1;
  int num_output_ = -1;
  int out_spatial_dim_ = -1;
  int top_count_ = -1;
  bool is_1x1_;



  int num_spatial_axes_ = -1;
  std::vector<int> kernel_shape_;
  std::vector<int> stride_;
  std::vector<int> pad_;
  std::vector<int> dilation_;
  std::vector<int> conv_input_shape_;
  std::vector<int> col_buffer_shape_;





protected:

  inline void conv_im2col(const DeviceTensor& data, DeviceTensor& col_buff) {
      im2col(data, conv_in_channels_,
          conv_input_shape_[1], conv_input_shape_[2],
          kernel_shape_[0], kernel_shape_[1],
          pad_[0], pad_[1],
          stride_[0], stride_[1],
          dilation_[0], dilation_[1], col_buff);
  }



  inline void conv_col2im(const DeviceTensor& col_buff, DeviceTensor& data) {
      col2im(col_buff, conv_in_channels_,
          conv_input_shape_[1], conv_input_shape_[2],
          kernel_shape_[0], kernel_shape_[1],
          pad_[0], pad_[1],
          stride_[0], stride_[1],
          dilation_[0], dilation_[1], data);
  }


  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int col_offset_;
  int output_offset_;

  DeviceTensor* col_buffer_;


};

 
}  // namespace hypertea

#endif  // HYPERTEA_BASE_CONVOLUTION_OP_HPP_
