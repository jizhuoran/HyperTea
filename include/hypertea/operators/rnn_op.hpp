#ifndef HYPERTEA_RNN_OP_HPP_
#define HYPERTEA_RNN_OP_HPP_

#include <vector>

#include "hypertea/operator.hpp"

namespace hypertea {

/**
 * @brief Rectified Linear Unit non-linearity @f$ y = \max(0, x) @f$.
 *        The simple max is fast to compute, and the function does not saturate.
 */



template <typename Dtype>
class CellParams
{
public:
  CellParams(
    const std::vector<TensorCPU<Dtype> >& _w_ih, 
    const std::vector<TensorCPU<Dtype> >& _w_hh, 
    const std::vector<TensorCPU<Dtype> >& _b_ih, 
    const std::vector<TensorCPU<Dtype> >& _b_hh
  ) : w_ih(_w_ih), w_hh(_w_hh), b_ih(_b_ih), b_hh(_b_hh) {};
  ~CellParams() {}

  const std::vector<TensorCPU<Dtype> >& w_ih;
  const std::vector<TensorCPU<Dtype> >& w_hh;
  const std::vector<TensorCPU<Dtype> >& b_ih; /* optional */
  const std::vector<TensorCPU<Dtype> >& b_hh; /* optional */


  // Tensor matmul_ih(Tensor input) const {
  //   return at::matmul(input, w_ih.t());
  // }
  // Tensor matmul_hh(Tensor h) const {
  //   return at::matmul(h, w_hh.t());
  // }
  std::vector<TensorCPU<Dtype> > linear_ih(TensorCPU<Dtype> input) const {

    std::vector<TensorCPU<Dtype> > result;

    for(int i = 0; i < w_ih.size(); ++i) {

      TensorCPU<Dtype> r(b_ih[i].duplicate_data(), b_ih[i].count());

      std::cout << "The shape is " << w_ih[i].shape()[0] << " and " << w_ih[i].shape()[1] << std::endl;


      hypertea_cpu_gemv<float>(CblasNoTrans, w_ih[i].shape()[0],
        w_ih[i].shape()[1], 1, w_ih[i].immutable_data(), input.immutable_data(),
        1, r.mutable_data());

      result.push_back(r);
    }

    return result;

  }


  std::vector<TensorCPU<Dtype> > linear_hh(TensorCPU<Dtype> input) const {

    std::vector<TensorCPU<Dtype> > result;

    for(int i = 0; i < w_hh.size(); ++i) {

      TensorCPU<Dtype> r(b_hh[i].duplicate_data(), b_hh[i].count());

      hypertea_cpu_gemv<float>(CblasNoTrans, w_hh[i].shape()[0],
        w_hh[i].shape()[1], 1, w_hh[i].immutable_data(), input.immutable_data(),
        1, r.mutable_data());

      result.push_back(r);
    }

    return result;

  }
  // Tensor linear_hh(Tensor h) const {
  //   return at::linear(h, w_hh, b_hh);
  // }

  
};

// struct CellParams {
  // CellParams(const Tensor& _w_ih, const Tensor& _w_hh, const Tensor& _b_ih, const Tensor& _b_hh)
    // : w_ih(_w_ih), w_hh(_w_hh), b_ih(_b_ih), b_hh(_b_hh) {};



  // const Tensor& w_ih;
  // const Tensor& w_hh;
  // const Tensor& b_ih; /* optional */
  // const Tensor& b_hh; /* optional */

  // Tensor matmul_ih(Tensor input) const {
  //   return at::matmul(input, w_ih.t());
  // }
  // Tensor matmul_hh(Tensor h) const {
  //   return at::matmul(h, w_hh.t());
  // }
  // Tensor linear_ih(Tensor input) const {
  //   return at::linear(input, w_ih, b_ih);
  // }
  // Tensor linear_hh(Tensor h) const {
  //   return at::linear(h, w_hh, b_hh);
  // }
// };


template <typename Dtype>
class Cell_CPU {
public:
  Cell_CPU() = default;
  ~Cell_CPU() = default;
  
  virtual TensorCPU<Dtype> Forward(TensorCPU<Dtype> &input_tensor, 
                                   TensorCPU<Dtype> &hidden_tensor,
                                   const CellParams<Dtype> & params);

};

template <typename Dtype>
class GRUCell_CPU : public Cell_CPU<Dtype> {
public:
  GRUCell_CPU() = default;
  ~GRUCell_CPU() = default;
  
  virtual TensorCPU<Dtype> Forward(TensorCPU<Dtype> &input_tensor, 
                                   TensorCPU<Dtype> &hidden_tensor,
                                   const CellParams<Dtype> & params);

};



// template <typename Dtype>
// class ReLUOp_CPU : public CPUFunctor<Dtype> {
//  public:
//   /**
//    * @param param provides ReLUParameter relu_param,
//    *     with ReLULayer options:
//    *   - negative_slope (\b optional, default 0).
//    *     the value @f$ \nu @f$ by which negative values are multiplied.
//    */
//   explicit ReLUOp_CPU(float negative_slope, bool inplace = false)
//       : CPUFunctor<Dtype>(), negative_slope_(negative_slope), inplace_(inplace) {}

//   virtual inline const char* type() const { return "ReLU"; }

//   // virtual void Forward(const std::vector<Dtype*> bottom_datas,
//   //     const std::vector<Dtype*> top_datas);

//   // virtual std::vector<Tensor<Dtype> *> Forward(const std::vector<Tensor<Dtype> *> inputs);
//   virtual TensorCPU<Dtype> Forward(TensorCPU<Dtype> &input_tensor);
  

//   private:
//     // int data_count_;
//     float negative_slope_;
//     bool inplace_;

// };

// #ifdef USE_OPENCL

// template <typename Dtype>
// class ReLUOp_GPU : public GPUFunctor<Dtype> {
//  public:
//   /**
//    * @param param provides ReLUParameter relu_param,
//    *     with ReLULayer options:
//    *   - negative_slope (\b optional, default 0).
//    *     the value @f$ \nu @f$ by which negative values are multiplied.
//    */
//   explicit ReLUOp_GPU(float negative_slope, bool inplace = false)
//       : GPUFunctor<Dtype>(), negative_slope_(negative_slope), inplace_(inplace) {}

//   virtual inline const char* type() const { return "ReLU"; }

//   // virtual void Forward(const std::vector<cl_mem> bottom_datas,
//   //     const std::vector<cl_mem> top_datas);

//   virtual TensorGPU<Dtype> Forward(TensorGPU<Dtype> input_tensor);


//   private:
//     float negative_slope_;
//     bool inplace_;

// };

// #endif //USE_OPENCL

}  // namespace hypertea

#endif  // HYPERTEA_RNN_OP_HPP_
