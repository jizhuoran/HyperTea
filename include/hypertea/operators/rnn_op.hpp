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


template <typename Dtype>
class Cell_CPU {
public:

  Cell_CPU() 
    :weight_ih_(nullptr),
    weight_hh_(nullptr),
    bias_ih_(nullptr),
    bias_hh_(nullptr) {}

  Cell_CPU(
      const int input_dim, const int hidden_dim,
      Dtype* weight_ih,
      Dtype* weight_hh,
      Dtype* bias_ih,
      Dtype* bias_hh
  ) : input_dim_(input_dim), hidden_dim_(hidden_dim),
      weight_ih_(weight_ih),
      weight_hh_(weight_hh),
      bias_ih_(bias_ih),
      bias_hh_(bias_hh) { }

  ~Cell_CPU() {

  }
  
  // virtual TensorCPU<Dtype> Forward(TensorCPU<Dtype> &input_tensor, 
  //                                  TensorCPU<Dtype> &hidden_tensor,
  //                                  const CellParams<Dtype> & params) {}

  virtual void Forward(
    Dtype* input_data,
    Dtype* hidden_data,
    Dtype* output_data
  ) { std::cout << "Why we come to this function" << std::endl;}

protected:

  int input_dim_, hidden_dim_;

  Dtype* weight_ih_;
  Dtype* weight_hh_;
  Dtype* bias_ih_;
  Dtype* bias_hh_;




};

template <typename Dtype>
class GRUCell_CPU : public Cell_CPU<Dtype> {
public:
  GRUCell_CPU(
      const int input_dim, const int hidden_dim,
      Dtype* weight_ih,
      Dtype* weight_hh,
      Dtype* bias_ih,
      Dtype* bias_hh) : 
        Cell_CPU<Dtype>(
          input_dim, hidden_dim, 
          weight_ih, weight_hh,
          bias_ih, bias_hh
        ) {
    
    intermediate_i = new Dtype[3 * this->hidden_dim_];
    intermediate_h = new Dtype[3 * this->hidden_dim_];

  }

  ~GRUCell_CPU() {
    delete [] intermediate_i;
    delete [] intermediate_h;
  }
  
  // virtual TensorCPU<Dtype> Forward(TensorCPU<Dtype> &input_tensor, 
  //                                  TensorCPU<Dtype> &hidden_tensor,
  //                                  const CellParams<Dtype> & params);


  virtual void Forward(
    Dtype* input_data,
    Dtype* hidden_data,
    Dtype* output_data
  );


private:
  Dtype* intermediate_i;
  Dtype* intermediate_h;


};




template <typename Dtype>
class RNNOp_CPU : public CPUFunctor<Dtype> {

public:
  RNNOp_CPU(
    int batch_size,
    int input_dim,
    int hidden_dim,
    Cell_CPU<Dtype>* cell) 
      : batch_size_(batch_size),
        input_dim_(input_dim), 
        hidden_dim_(hidden_dim),
        cell_(cell) {}

  ~RNNOp_CPU()  {
    delete cell_;
  }

  
  virtual TensorCPU<Dtype> Forward(TensorCPU<Dtype> &input_tensor, TensorCPU<Dtype> &hidden_tensor) = 0;

private:

  int batch_size_;
  int input_dim_, hidden_dim_;

  Cell_CPU<Dtype>* cell_;

};





template <typename Dtype>
class UnidirectionalRNN_CPU : public RNNOp_CPU<Dtype> {

}


// template <typename Dtype>
// class GRUOp_CPU : public RNNOp_CPU<Dtype> {
// public:
//   GRUOp_CPU(
//     int batch_size,
//     int input_dim,
//     int hidden_dim,
//     Dtype* const weight_ih,
//     Dtype* const weight_hh,
//     Dtype* const bias_ih,
//     Dtype* const bias_hh) 
//       : RNNOp_CPU<Dtype>(batch_size, input_dim, hidden_dim, 
//                          new GRUCell_CPU<Dtype>(
//                               input_dim, hidden_dim, 
//                               weight_ih, weight_hh,
//                               bias_ih, bias_hh)
//                         ) { }
//   ~GRUOp_CPU() = default;



//   virtual inline const char* type() const { return "GRU"; }

//   // virtual void Forward(const std::vector<Dtype*> bottom_datas,
//       // const std::vector<Dtype*> top_datas);
//   virtual TensorCPU<Dtype> Forward(TensorCPU<Dtype> &input_tensor) {
//     auto hidden_tensor = TensorCPU<Dtype>(this->hidden_dim_, Dtype(.0));
//     Forward(input_tensor, hidden_tensor);
//   }
  
//   TensorCPU<Dtype> Forward(TensorCPU<Dtype> &input_tensor, TensorCPU<Dtype> &hidden_tensor);

// };




}  // namespace hypertea

#endif  // HYPERTEA_RNN_OP_HPP_
