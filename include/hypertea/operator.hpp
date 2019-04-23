#ifndef HYPERTEA_LAYER_H_
#define HYPERTEA_LAYER_H_

#include "hypertea/tensor.hpp"

namespace hypertea {

// template <
//   typename DeviceTensor, 
//   typename = typename std::enable_if<
//       std::is_base_of<Tensor<float>, 
//         typename std::decay<DeviceTensor>::type
//       >::value 
//       || 
//       std::is_base_of<Tensor<half>, 
//         typename std::decay<DeviceTensor>::type
//       >::value
//   >::type
// >

template <typename DeviceTensor>
class TensorOperator {

public:

  explicit TensorOperator() {}
  virtual ~TensorOperator() {}
  
  virtual inline const char* type() const = 0;
  virtual DeviceTensor operator()(DeviceTensor input) = 0;


};  



}  // namespace hypertea

#endif  // HYPERTEA_LAYER_H_
