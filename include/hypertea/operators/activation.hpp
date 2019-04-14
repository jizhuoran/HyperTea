#ifndef HYPERTEA_ACTIVATION_OP_HPP_
#define HYPERTEA_ACTIVATION_OP_HPP_

#include "hypertea/operator.hpp"

namespace hypertea {


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

}  // namespace hypertea

#endif  // HYPERTEA_ACTIVATION_OP_HPP_
