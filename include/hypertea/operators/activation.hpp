#ifndef HYPERTEA_ACTIVATION_OP_HPP_
#define HYPERTEA_ACTIVATION_OP_HPP_

#include "hypertea/operator.hpp"

namespace hypertea {

template <typename Dtype>
TensorCPU<Dtype> inplace_prelu(
    TensorCPU<Dtype> x, 
    const TensorCPU<Dtype>& weight,
    int channels, int spatial_dim
) {

    int index;

    auto weight_data = weight.immutable_data();
    DEFINE_VSL_CHANNEL_FUNC(
        index = (n * channels + c) * spatial_dim + i; 
        data[index] = std::max(data[index], float(0)) + weight_data[c] * std::min(data[index], float(0))
    );
    return x;
}
 
template <typename Dtype>
TensorGPU<Dtype> inplace_prelu(
  TensorGPU<Dtype> x, 
  const TensorGPU<Dtype>& weight,
  int channels, int inner_dim) {
  

  int N = x.count();
  auto data = x.mutable_data();
  auto weight_ = weight.mutable_data();

 
  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    "prelu_kernel",
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&data),
      std::make_pair(sizeof(cl_mem), (void *)&data),
      std::make_pair(sizeof(cl_int), (void *)&N),
      std::make_pair(sizeof(cl_mem), (void *)&weight_),
      std::make_pair(sizeof(cl_int), (void *)&channels),
      std::make_pair(sizeof(cl_int), (void *)&inner_dim),

    },
    std::vector<size_t> {HYPERTEA_GET_BLOCKS(N)},
    std::vector<size_t> {HYPERTEA_OPENCL_NUM_THREADS}
  );

  return x;

}


template TensorGPU<float> inplace_prelu(
  TensorGPU<float> x, 
  const TensorGPU<float>& weight,
  int channels,
  int inner_dim
);

template TensorGPU<half> inplace_prelu(
  TensorGPU<half> x, 
  const TensorGPU<half>& weight,
  int channels,
  int inner_dim
);


template <typename DeviceTensor>
class PReLUOp : public TensorOperator<DeviceTensor>{

public:
    explicit PReLUOp(
        DeviceTensor* weight, 
        int channels,
	    int inner_dim,
        bool inplace = false) 
    : TensorOperator<DeviceTensor>(), 
    weight_(weight), 
    channels_(channels),
    inner_dim_(inner_dim),
    inplace_(inplace) {}
    
    virtual inline const char* type() const override { return "PReLU"; }
    virtual DeviceTensor operator()(DeviceTensor input) override;

private:
    DeviceTensor* weight_;
    int channels_;
	int inner_dim_;
    bool inplace_;

}; 

template <typename DeviceTensor>
class ReLUOp : public TensorOperator<DeviceTensor>{

public:
    explicit ReLUOp(float negative_slope, bool inplace = false) 
    : TensorOperator<DeviceTensor>(), negative_slope_(negative_slope), inplace_(inplace) {}
    
    virtual inline const char* type() const override { return "ReLU"; }
    virtual DeviceTensor operator()(DeviceTensor input) override;

private:
    float negative_slope_;
    bool inplace_;

};


template <typename DeviceTensor>
class TanHOp : public TensorOperator<DeviceTensor>{

public:
    explicit TanHOp(bool inplace = false)
    : TensorOperator<DeviceTensor>(), inplace_(inplace) {}

    virtual inline const char* type() const override { return "TanH"; }
    virtual DeviceTensor operator()(DeviceTensor input) override;

private:
    bool inplace_;

};


template <typename DeviceTensor>
class ELUOp : public TensorOperator<DeviceTensor>{

public:

    explicit ELUOp(float alpha, bool inplace = false)
    : TensorOperator<DeviceTensor>(), alpha_(alpha), inplace_(inplace) {}

    virtual inline const char* type() const override { return "ELU"; }
    virtual DeviceTensor operator()(DeviceTensor input) override;

private:
    float alpha_;
    bool inplace_;

};


template <typename DeviceTensor>
class SoftMaxOp : public TensorOperator<DeviceTensor>{

public:

    explicit SoftMaxOp(int spatial_dim, bool inplace = false)
    : TensorOperator<DeviceTensor>(), spatial_dim_(spatial_dim), inplace_(inplace) {}

    virtual inline const char* type() const override { return "SoftMax"; }
    virtual DeviceTensor operator()(DeviceTensor input) override;

private:
    int spatial_dim_;
    bool inplace_;

};

}  // namespace hypertea

#endif  // HYPERTEA_ACTIVATION_OP_HPP_
