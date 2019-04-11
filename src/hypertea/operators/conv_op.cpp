#include <vector>

#include "hypertea/operators/conv_op.hpp"
// #include "hypertea/util/im2col.hpp"

namespace hypertea {


template<typename DeviceTensor>
DeviceTensor ConvolutionOp<DeviceTensor>::operator()(DeviceTensor& input) {
  
  auto output = DeviceTensor(this->top_count_);
  output.set(0);


  auto inputs_tensors  = input.chunked_tensors(this->num_);
  auto outputs_tensors = output.chunked_tensors(this->num_);

  for (int i = 0; i < this->num_; ++i) {


    if (!this->is_1x1_) {
      this->conv_im2col(inputs_tensors[i], *this->col_buffer_);
    } else {
      this->col_buffer_ = &inputs_tensors[i];
    }

    inplace_gemm(
      CblasNoTrans, CblasNoTrans, 
      this->conv_out_channels_, this->conv_out_spatial_dim_, this->kernel_dim_,
      (float)1., *this->weight_, *this->col_buffer_,
      (float)0., outputs_tensors[i]
    );

    if (this->bias_) {
      inplace_channeled_add(outputs_tensors[i], *this->bias_, this->num_output_, this->out_spatial_dim_);
    }

  }

  return output;

}
DEFINE_FORWARD_FUNC(ConvolutionOp);


}  // namespace hypertea
