// hypertea.hpp is the header file that you need to include in your code. It wraps
// all the internal hypertea header files into one for simpler inclusion.

#ifndef HYPERTEA_HYPERTEA_HPP_
#define HYPERTEA_HYPERTEA_HPP_

#include "hypertea/common.hpp"


#include "hypertea/operators/elu_op.hpp"
#include "hypertea/operators/relu_op.hpp"
#include "hypertea/operators/conv_op.hpp"
#include "hypertea/operators/deconv_op.hpp"
#include "hypertea/operators/eltwise_op.hpp"
#include "hypertea/operators/scale_op.hpp"
#include "hypertea/operators/split_op.hpp"
#include "hypertea/operators/tanh_op.hpp"
#include "hypertea/operators/batch_norm_op.hpp"

#endif  // HYPERTEA_HYPERTEA_HPP_