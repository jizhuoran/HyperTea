#ifndef HYPERTEA_SAMPLING_OP_HPP_
#define HYPERTEA_SAMPLING_OP_HPP_

#include "hypertea/operator.hpp"

namespace hypertea {


template <typename DeviceTensor>
class UpSampling2D : public TensorOperator<DeviceTensor>{

public:
    explicit UpSampling2D(int scale, int width, int height) 
    : TensorOperator<DeviceTensor>(), scale_(scale), width_(width), height_(height) {}
    
    virtual inline const char* type() const override { return "UpSampling2D"; }
    virtual DeviceTensor operator()(DeviceTensor input) override;

private:
    
    int scale_;
    int width_;
    int height_;

};



}  // namespace hypertea

#endif  // HYPERTEA_SAMPLING_OP_HPP_
