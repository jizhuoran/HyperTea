#ifndef HYPERTEA_DECONV_OP_HPP_
#define HYPERTEA_DECONV_OP_HPP_

#include <vector>

#include "hypertea/operators/base_conv_op.hpp"

namespace hypertea {

template <typename DeviceTensor>
class DeconvolutionOp : public BaseConvolutionOp<DeviceTensor> {
 public:

  explicit DeconvolutionOp(
    DeviceTensor* weight, 
    DeviceTensor* bias,
    int group,
    bool is_1x1,
    std::vector<int> kernel_shape,
    std::vector<int> stride,
    std::vector<int> pad,
    std::vector<int> dilation,
    std::vector<int> input_shape,
    std::vector<int> output_shape) 

    : BaseConvolutionOp<DeviceTensor>(weight, bias, group, is_1x1,
      kernel_shape, stride, pad, dilation, input_shape, output_shape, true) {}



  inline const char* type() const { return "Deconvolution"; }

  DeviceTensor operator()(DeviceTensor &input);

};



}  // namespace hypertea

#endif  // HYPERTEA_DECONV_OP_HPP_
