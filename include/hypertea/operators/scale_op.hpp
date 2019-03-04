#ifndef HYPERTEA_SCALE_OP_HPP_
#define HYPERTEA_SCALE_OP_HPP_

#include <vector>

#include "hypertea/operator.hpp"


namespace hypertea {

/**
 * @brief Computes the elementwise product of two input Blobs, with the shape of
 *        the latter Blob "broadcast" to match the shape of the former.
 *        Equivalent to tiling the latter Blob, then computing the elementwise
 *        product. Note: for efficiency and convenience, this layer can
 *        additionally perform a "broadcast" sum too when `bias_term: true`
 *        is set.
 *
 * The latter, scale input may be omitted, in which case it's learned as
 * parameter of the layer (as is the bias, if it is included).
 */
template <typename Dtype>
class ScaleOp_CPU: public CPUFunctor<Dtype> {
 public:
  explicit ScaleOp_CPU(Dtype* bias_data, Dtype* scale_data, 
                   int scale_dim, int outer_dim, int inner_dim, bool inplace = false)
      : CPUFunctor<Dtype>(),
        bias_data_(bias_data), scale_data_(scale_data), 
        scale_dim_(scale_dim), outer_dim_(outer_dim), 
        inner_dim_(inner_dim), inplace_(inplace) {

          if (bias_data != NULL) {
            bias_multiplier_ = new Dtype[inner_dim_];
            hypertea_set(inner_dim_, Dtype(1), bias_multiplier_);
          }
          
        }

  ~ScaleOp_CPU() {
    
    if (bias_multiplier_ != NULL) {
      delete [] bias_multiplier_;
    }
  }

  virtual inline const char* type() const { return "Scale"; }

  virtual TensorCPU<Dtype> Forward(TensorCPU<Dtype> &input_tensor);


private:

  bool inplace_;


  Dtype* bias_data_;
  Dtype* scale_data_;
  
  Dtype* bias_multiplier_ = NULL;


  int outer_dim_, scale_dim_, inner_dim_;

};

#ifdef USE_OPENCL

template <typename Dtype>
class ScaleOp_GPU: public GPUFunctor<Dtype> {
 public:

  explicit ScaleOp_GPU(int top_count,
                   cl_mem bias_data, cl_mem scale_data, 
                   int scale_dim, int inner_dim)
      : GPUFunctor<Dtype>(), top_count_(top_count), 
        bias_data_(bias_data), scale_data_(scale_data), 
        scale_dim_(scale_dim), inner_dim_(inner_dim) {}


  virtual inline const char* type() const { return "Scale"; }

  // virtual void Forward(const std::vector<cl_mem> bottom_datas,
  //     const std::vector<cl_mem> top_datas);

  virtual TensorGPU<Dtype> Forward(TensorGPU<Dtype> input_tensor);
  


  int top_count_;

  cl_mem bias_data_;
  cl_mem scale_data_;
  
  int scale_dim_, inner_dim_;

  bool inplace_ = false;

};


#endif //USE_OPENCL


}  // namespace hypertea

#endif  // HYPERTEA_SCALE_OP_HPP_
