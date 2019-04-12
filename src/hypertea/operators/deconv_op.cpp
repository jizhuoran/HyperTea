#include <vector>

#include "hypertea/operators/deconv_op.hpp"

namespace hypertea {



template<typename DeviceTensor>
DeviceTensor DeconvolutionOp<DeviceTensor>::operator()(DeviceTensor input) {


  auto output = DeviceTensor(this->top_count_, 0);

  auto inputs_tensors  = input.chunked_tensors(this->num_);
  auto outputs_tensors = output.chunked_tensors(this->num_);

  for (int i = 0; i < this->num_; ++i) {

    if (this->is_1x1_) {
      this->col_buffer_ = &outputs_tensors[i];
    }

    inplace_gemm(
      CblasTrans, CblasNoTrans, 
      this->kernel_dim_, this->conv_out_spatial_dim_, this->conv_out_channels_,
      (float)1., *this->weight_, inputs_tensors[i],
      (float)0., *this->col_buffer_
    );

    if (!this->is_1x1_) {
      this->conv_col2im(*this->col_buffer_, outputs_tensors[i]);
    }

    if (this->bias_) {
      inplace_channeled_add(outputs_tensors[i], *this->bias_, this->num_output_, this->out_spatial_dim_);
    }

  }

  return output;

}


DEFINE_FORWARD_FUNC(DeconvolutionOp);


}  // namespace hypertea
