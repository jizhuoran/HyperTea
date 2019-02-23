#ifndef HYPERTEA_CONV_OP_HPP_
#define HYPERTEA_CONV_OP_HPP_

#include <vector>

#include "hypertea/operators/base_conv_op.hpp"

namespace hypertea {

/**
 * @brief Convolves the input image with a bank of learned filters,
 *        and (optionally) adds biases.
 *
 *   Hypertea convolves by reduction to matrix multiplication. This achieves
 *   high-throughput and generality of input and filter dimensions but comes at
 *   the cost of memory for matrices. This makes use of efficiency in BLAS.
 *
 *   The input is "im2col" transformed to a channel K' x H x W data matrix
 *   for multiplication with the N x K' x H x W filter matrix to yield a
 *   N' x H x W output matrix that is then "col2im" restored. K' is the
 *   input channel * kernel height * kernel width dimension of the unrolled
 *   inputs so that the im2col matrix has a column for each input region to
 *   be filtered. col2im restores the output spatial structure by rolling up
 *   the output channel N' columns of the output matrix.
 */
template <typename Dtype>
class ConvolutionOp_CPU : public BaseConvolutionOp_CPU<Dtype> {
 public:

  explicit ConvolutionOp_CPU(const Dtype* weight, const Dtype* bias,
              int group,
              bool is_1x1,
              std::vector<int> kernel_shape,
              std::vector<int> stride,
              std::vector<int> pad,
              std::vector<int> dilation,
              std::vector<int> input_shape,
              std::vector<int> output_shape,
              bool force_nd_im2col) 

    : BaseConvolutionOp_CPU<Dtype>(weight, bias, group, is_1x1,
      kernel_shape, stride, pad, dilation, input_shape, output_shape, force_nd_im2col, false) {}



  virtual inline const char* type() const { return "Convolution"; }

 // protected:
  virtual void Forward(const std::vector<Dtype*> bottom_datas,
      const std::vector<Dtype*> top_datas);

  virtual std::vector<Tensor<Dtype> *> Forward(const std::vector<Tensor<Dtype> *> inputs);

  
};

#ifdef USE_OPENCL

template <typename Dtype>
class ConvolutionOp_GPU : public BaseConvolutionOp_GPU<Dtype> {
 public:

  explicit ConvolutionOp_GPU(std::string kernel_name, int bottom_size,
                             cl_mem weight, cl_mem bias,
                             std::vector<int> local,
                             std::vector<int> global)

      : BaseConvolutionOp_GPU<Dtype>(kernel_name, bottom_size, weight, bias,
                                 local, global) { }

  virtual inline const char* type() const { return "Convolution"; }

  virtual void Forward(const std::vector<cl_mem> bottom_datas,
      const std::vector<cl_mem> top_datas);
  
};

#endif //USE_OPENCL


}  // namespace hypertea

#endif  // HYPERTEA_CONV_OP_HPP_
