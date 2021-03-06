#ifndef HYPERTEA_LINEAR_OP_HPP_
#define HYPERTEA_LINEAR_OP_HPP_

#include "hypertea/operator.hpp"

namespace hypertea {


template <typename DeviceTensor>
class LinearOp : public TensorOperator<DeviceTensor>{

public:
    explicit LinearOp(
        DeviceTensor* weight,
        DeviceTensor* bias,
        int in_features,
	    int out_features) 
    : TensorOperator<DeviceTensor>(), 
    weight_(weight), 
    bias_(bias),
    in_features_(in_features),
    out_features_(out_features) {}
    
    virtual inline const char* type() const override { return "Linear"; }
    virtual DeviceTensor operator()(DeviceTensor input) override;

private:
    DeviceTensor* weight_;
    DeviceTensor* bias_;
	int in_features_;
    int out_features_;

};



template <typename DeviceTensor>
class EmbeddingOp : public TensorOperator<DeviceTensor>{

public:
    explicit EmbeddingOp(
        DeviceTensor* weight,
        int embedding_dim) 
    : TensorOperator<DeviceTensor>(), 
    weight_(weight), 
    embedding_dim_(embedding_dim) {}
    
    virtual inline const char* type() const override { return "Embedding"; }
    virtual DeviceTensor operator()(DeviceTensor input) override {};

    DeviceTensor operator()(std::vector<int> input);

private:
    DeviceTensor* weight_;
    DeviceTensor* bias_;
    int embedding_dim_;

};


}  // namespace hypertea

#endif  // HYPERTEA_LINEAR_OP_HPP_
