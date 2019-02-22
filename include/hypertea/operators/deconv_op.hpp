#ifndef HYPERTEA_DECONV_LAYER_HPP_
#define HYPERTEA_DECONV_LAYER_HPP_

#include <vector>

#include "hypertea/operators/base_conv_op.hpp"

namespace hypertea {

/**
 * @brief Convolve the input with a bank of learned filters, and (optionally)
 *        add biases, treating filters and convolution parameters in the
 *        opposite sense as ConvolutionLayer.
 *
 *   ConvolutionLayer computes each output value by dotting an input window with
 *   a filter; DeconvolutionLayer multiplies each input value by a filter
 *   elementwise, and sums over the resulting output windows. In other words,
 *   DeconvolutionLayer is ConvolutionLayer with the forward and backward passes
 *   reversed. DeconvolutionLayer reuses ConvolutionParameter for its
 *   parameters, but they take the opposite sense as in ConvolutionLayer (so
 *   padding is removed from the output rather than added to the input, and
 *   stride results in upsampling rather than downsampling).
 */
template <typename Dtype>
class DeconvolutionOp_CPU : public BaseConvolutionOp_CPU<Dtype> {
 public:

  explicit DeconvolutionOp_CPU(const Dtype* weight, const Dtype* bias,
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

    : BaseConvolutionOp_CPU<Dtype>(weight, bias, bottom_dim, bottom_size, top_dim, num, channels, 
      group, weight_offset, num_output, out_spatial_dim, is_1x1, force_nd_im2col, conv_out_channels, 
      conv_in_channels, conv_out_spatial_dim, kernel_dim, col_offset, output_offset, 
      num_spatial_axes, kernel_shape, stride, pad, dilation, conv_input_shape, col_buffer_shape) {}

  virtual inline const char* type() const { return "Deconvolution"; }

 // protected:
  virtual void Forward(const std::vector<Dtype*> bottom_datas,
      const std::vector<Dtype*> top_datas);

  virtual std::vector<Tensor<Dtype> *> Forward(const std::vector<Tensor<Dtype> *> inputs);

};

#ifdef USE_OPENCL

template <typename Dtype>
class DeconvolutionOp_GPU : public BaseConvolutionOp_GPU<Dtype> {
 public:

  explicit DeconvolutionOp_GPU(std::string kernel_name, int bottom_size,
                             cl_mem weight, cl_mem bias,
                             std::vector<int> local,
                             std::vector<int> global)

      : BaseConvolutionOp_GPU<Dtype>(kernel_name, bottom_size, weight, bias,
                                 local, global) { }

  virtual inline const char* type() const { return "Deconvolution"; }

 // protected:
  virtual void Forward(const std::vector<cl_mem> bottom_datas,
      const std::vector<cl_mem> top_datas);
};

#endif //USE_OPENCL


}  // namespace hypertea

#endif  // HYPERTEA_DECONV_LAYER_HPP_
