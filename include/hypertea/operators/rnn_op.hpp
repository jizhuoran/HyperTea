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
class Cell_CPU {
public:


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
    delete [] intermediate_i;
    delete [] intermediate_h;
  }
  
  virtual void Forward(
    Dtype* input_data,
    Dtype* hidden_data,
    Dtype* output_data
  ) = 0;


  virtual int hidden_offset_() = 0;


protected:

  int input_dim_, hidden_dim_;

  Dtype* weight_ih_;
  Dtype* weight_hh_;
  Dtype* bias_ih_;
  Dtype* bias_hh_;

  Dtype* intermediate_i;
  Dtype* intermediate_h;


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
    
    this->intermediate_i = new Dtype[3 * this->hidden_dim_];
    this->intermediate_h = new Dtype[3 * this->hidden_dim_];

  }

  ~GRUCell_CPU() {}
  
  virtual void Forward(
    Dtype* input_data,
    Dtype* hidden_data,
    Dtype* output_data
  );
  
  virtual int hidden_offset_() {return this->hidden_dim_;}
  

};

template <typename Dtype>
class LSTMCell_CPU : public Cell_CPU<Dtype> {
public:
  LSTMCell_CPU(
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
    
    this->intermediate_i = new Dtype[4 * this->hidden_dim_];
    this->intermediate_h = new Dtype[4 * this->hidden_dim_];

  }

  ~LSTMCell_CPU() {}
  
  virtual void Forward(
    Dtype* input_data,
    Dtype* hidden_data,
    Dtype* output_data
  );

  virtual int hidden_offset_() {return 2 * this->hidden_dim_;}


};





template <typename Dtype>
class RNNOp_CPU {

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
    // delete cell_;
  }

  
  virtual TensorCPU<Dtype> Forward(TensorCPU<Dtype> &input_tensor, TensorCPU<Dtype> &hidden_tensor) = 0;
  

protected:

  int batch_size_;
  int input_dim_, hidden_dim_;

  Cell_CPU<Dtype>* cell_;

};






template <typename Dtype>
class UnidirectionalRNN_CPU : public RNNOp_CPU<Dtype> {

public:
  UnidirectionalRNN_CPU(
    int batch_size,
    int input_dim,
    int hidden_dim,
    Cell_CPU<Dtype>* cell) 
      : RNNOp_CPU<Dtype>(batch_size, input_dim, hidden_dim, cell) {}

  ~UnidirectionalRNN_CPU() {}


  virtual TensorCPU<Dtype> Forward(TensorCPU<Dtype> &input_tensor, TensorCPU<Dtype> &hidden_tensor);

};


template <typename Dtype>
class BidirectionalRNN_CPU : public RNNOp_CPU<Dtype> {

public:
  BidirectionalRNN_CPU(
    int batch_size,
    int input_dim,
    int hidden_dim,
    Cell_CPU<Dtype>* cell, 
    Cell_CPU<Dtype>* reverse_cell) 
      : RNNOp_CPU<Dtype>(batch_size, input_dim, hidden_dim, cell), reverse_cell_(reverse_cell) {}

  ~BidirectionalRNN_CPU() {}


  virtual TensorCPU<Dtype> Forward(TensorCPU<Dtype> &input_tensor, TensorCPU<Dtype> &hidden_tensor);

private:
  
  Cell_CPU<Dtype>* reverse_cell_;


};


template <typename Dtype>
class StackedRNN : public CPUFunctor<Dtype> {

public:
  StackedRNN(
    // int batch_size,
    // int input_dim,
    // int hidden_dim,
    std::vector<RNNOp_CPU<Dtype>* > rnn_layers) 
      : rnn_layers_(rnn_layers) {}

  ~StackedRNN()  {}

  
  virtual TensorCPU<Dtype> Forward(TensorCPU<Dtype> input_tensor, 
    std::vector<TensorCPU<Dtype> > hidden_tensors);
  
  virtual TensorCPU<Dtype> Forward(TensorCPU<Dtype> &input_tensor) { }

protected:

  int batch_size_;
  int input_dim_, hidden_dim_;

  std::vector<RNNOp_CPU<Dtype>* > rnn_layers_;


};





}  // namespace hypertea

#endif  // HYPERTEA_RNN_OP_HPP_
