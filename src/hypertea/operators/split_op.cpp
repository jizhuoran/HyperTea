#include <vector>

#include "hypertea/operators/split_op.hpp"
#include "hypertea/util/math_functions.hpp"

namespace hypertea {


template <typename Dtype>
void SplitOp_CPU<Dtype>::Forward(const Dtype* bottom_data,
      const std::vector<Dtype*> top_datas) {
  for (int i = 0; i < top_datas.size(); ++i) {    
    hypertea_copy(data_count_, bottom_data, top_datas[i]);
  }
}

#ifdef USE_OPENCL

template <typename Dtype>
void SplitOp_GPU<Dtype>::Forward(const cl_mem bottom_data,
      const std::vector<cl_mem> top_datas) {

  for (int i = 0; i < top_datas.size(); ++i) {
    hypertea_cl_copy<Dtype>(data_count_, bottom_data, top_datas[i]);
  }
}
#endif //USE_OPENCL


INSTANTIATE_CLASS_CPU(SplitOp_CPU);
INSTANTIATE_CLASS_GPU(SplitOp_GPU);
// REGISTER_LAYER_CLASS(Split);

}  // namespace hypertea
