#ifndef HYPERTEA_RNN_OP_HPP_
#define HYPERTEA_RNN_OP_HPP_

#include <vector>

#include "hypertea/operator.hpp"

namespace hypertea {

/**
 * @brief Rectified Linear Unit non-linearity @f$ y = \max(0, x) @f$.
 *        The simple max is fast to compute, and the function does not saturate.
 */


enum RNN_CELL_TYPE {
  LSTM_CELL,
  GRU_CELL
};



template <typename Dtype>
class Cell_CPU {
public:


  Cell_CPU(
      const int input_dim, const int hidden_dim,
      const TensorCPU<Dtype>& weight_ih,
      const TensorCPU<Dtype>& weight_hh,
      const TensorCPU<Dtype>& bias_ih,
      const TensorCPU<Dtype>& bias_hh,
      const TensorCPU<Dtype>& inter_i,
      const TensorCPU<Dtype>& inter_h
  ) : input_dim_(input_dim), hidden_dim_(hidden_dim),
      weight_ih_(weight_ih),
      weight_hh_(weight_hh),
      bias_ih_(bias_ih),
      bias_hh_(bias_hh),
      intermediate_i(inter_i),
      intermediate_h(inter_h) { }

  virtual ~Cell_CPU() {}
  
  virtual void Forward(
    TensorCPU<Dtype>& input_data,
    TensorCPU<Dtype>& hidden_data,
    TensorCPU<Dtype>& output_data
  ) = 0;


  virtual int hidden_offset_() = 0;


protected:

  int input_dim_, hidden_dim_;

  TensorCPU<Dtype> weight_ih_;
  TensorCPU<Dtype> weight_hh_;
  TensorCPU<Dtype> bias_ih_;
  TensorCPU<Dtype> bias_hh_;

  TensorCPU<Dtype> intermediate_i;
  TensorCPU<Dtype> intermediate_h;


};





template <typename Dtype>
class GRUCell_CPU : public Cell_CPU<Dtype> {
public:
  GRUCell_CPU(
      const int input_dim, const int hidden_dim,
      const TensorCPU<Dtype>& weight_ih,
      const TensorCPU<Dtype>& weight_hh,
      const TensorCPU<Dtype>& bias_ih,
      const TensorCPU<Dtype>& bias_hh) : 
        Cell_CPU<Dtype>(
          input_dim, hidden_dim, 
          weight_ih, weight_hh,
          bias_ih, bias_hh,
          TensorCPU<Dtype>(3 * hidden_dim),
          TensorCPU<Dtype>(3 * hidden_dim)
        ) {}

  virtual ~GRUCell_CPU() {}
  
  virtual void Forward(
    TensorCPU<Dtype>& input_data,
    TensorCPU<Dtype>& hidden_data,
    TensorCPU<Dtype>& output_data
  );
  
  virtual int hidden_offset_() {return this->hidden_dim_;}
  

};

template <typename Dtype>
class LSTMCell_CPU : public Cell_CPU<Dtype> {
public:
  LSTMCell_CPU(
      const int input_dim, const int hidden_dim,
      const TensorCPU<Dtype>& weight_ih,
      const TensorCPU<Dtype>& weight_hh,
      const TensorCPU<Dtype>& bias_ih,
      const TensorCPU<Dtype>& bias_hh) : 
        Cell_CPU<Dtype>(
          input_dim, hidden_dim, 
          weight_ih, weight_hh,
          bias_ih, bias_hh,
          TensorCPU<Dtype>(4 * hidden_dim),
          TensorCPU<Dtype>(4 * hidden_dim)
        ) { }

  virtual ~LSTMCell_CPU() {}
  
  virtual void Forward(
    TensorCPU<Dtype>& input_data,
    TensorCPU<Dtype>& hidden_data,
    TensorCPU<Dtype>& output_data
  );

  virtual int hidden_offset_() {return 2 * this->hidden_dim_;}


};

template <typename Dtype>
Cell_CPU<Dtype>* cell_factory_cpu_(
    const int input_dim, 
    const int hidden_dim,
    const TensorCPU<Dtype>& w_ih,
    const TensorCPU<Dtype>& w_hh,
    const TensorCPU<Dtype>& b_ih,
    const TensorCPU<Dtype>& b_hh,
    RNN_CELL_TYPE cell_type) {

  switch (cell_type) {
    case RNN_CELL_TYPE::GRU_CELL: {
      return new hypertea::GRUCell_CPU<Dtype>(input_dim, hidden_dim, w_ih, w_hh, b_ih, b_hh);
    }
    case RNN_CELL_TYPE::LSTM_CELL: {
      return new hypertea::LSTMCell_CPU<Dtype>(input_dim, hidden_dim, w_ih, w_hh, b_ih, b_hh);
    }
    default: {
      std::cout << "Wrong RNN Cell Type!" << std::endl;
      exit(0);
    }
  }

}





template <typename Dtype>
class RNNOp_CPU {

public:
  RNNOp_CPU(
    int input_dim,
    int hidden_dim,
    const TensorCPU<Dtype>& w_ih,
    const TensorCPU<Dtype>& w_hh,
    const TensorCPU<Dtype>& b_ih,
    const TensorCPU<Dtype>& b_hh,
    RNN_CELL_TYPE cell_type) 
      : input_dim_(input_dim), 
        hidden_dim_(hidden_dim),
        cell_(cell_factory_cpu_<Dtype>(input_dim, hidden_dim, w_ih, w_hh, b_ih, b_hh, cell_type)) {}

  ~RNNOp_CPU()  {}

  
  virtual TensorCPU<Dtype> Forward(TensorCPU<Dtype> &input_tensor, TensorCPU<Dtype> &hidden_tensor) = 0;
  

protected:

  int batch_size_ = 1;
  int input_dim_, hidden_dim_;

  std::unique_ptr<Cell_CPU<Dtype>> cell_;


};






template <typename Dtype>
class UnidirectionalRNN_CPU : public RNNOp_CPU<Dtype> {

public:
  UnidirectionalRNN_CPU(
    int input_dim,
    int hidden_dim,
    const TensorCPU<Dtype>& w_ih,
    const TensorCPU<Dtype>& w_hh,
    const TensorCPU<Dtype>& b_ih,
    const TensorCPU<Dtype>& b_hh,
    RNN_CELL_TYPE cell_type) 
      : RNNOp_CPU<Dtype>(input_dim, hidden_dim, w_ih, w_hh, b_ih, b_hh, cell_type) {}

  ~UnidirectionalRNN_CPU() {}


  virtual TensorCPU<Dtype> Forward(TensorCPU<Dtype> &input_tensor, TensorCPU<Dtype> &hidden_tensor);

};


template <typename Dtype>
class BidirectionalRNN_CPU : public RNNOp_CPU<Dtype> {

public:
  BidirectionalRNN_CPU(
    int input_dim,
    int hidden_dim,
    const TensorCPU<Dtype>& w_ih, const TensorCPU<Dtype>& rw_ih,
    const TensorCPU<Dtype>& w_hh, const TensorCPU<Dtype>& rw_hh,
    const TensorCPU<Dtype>& b_ih, const TensorCPU<Dtype>& rb_ih,
    const TensorCPU<Dtype>& b_hh, const TensorCPU<Dtype>& rb_hh,
    RNN_CELL_TYPE cell_type) 
      : RNNOp_CPU<Dtype>(input_dim, hidden_dim, w_ih, w_hh, b_ih, b_hh, cell_type),
        reverse_cell_(cell_factory_cpu_<Dtype>(input_dim, hidden_dim, rw_ih, rw_hh, rb_ih, rb_hh, cell_type)) { }

  ~BidirectionalRNN_CPU() {}


  virtual TensorCPU<Dtype> Forward(TensorCPU<Dtype> &input_tensor, TensorCPU<Dtype> &hidden_tensor);

private:
  
  std::unique_ptr<Cell_CPU<Dtype>> reverse_cell_;


};


template <typename Dtype>
class StackedRNN_CPU : public CPUFunctor<Dtype> {

public: 
  StackedRNN_CPU(
    std::vector<RNNOp_CPU<Dtype>* > rnn_layers) 
      : rnn_layers_(rnn_layers) {}

  ~StackedRNN_CPU()  {
    for (int i = 0; i < rnn_layers_.size(); ++i) {
      delete rnn_layers_[i];
    }
  }

  
  virtual TensorCPU<Dtype> Forward(TensorCPU<Dtype> input_tensor, 
    std::vector<TensorCPU<Dtype> > hidden_tensors);
  
  virtual TensorCPU<Dtype> Forward(TensorCPU<Dtype> &input_tensor) { }

private:

  std::vector<RNNOp_CPU<Dtype>* > rnn_layers_;


};





////// TODO //////

template <typename Dtype>
class Cell_GPU {
public:


  Cell_GPU(
      const int input_dim, const int hidden_dim,
      TensorGPU<Dtype> weight_ih,
      TensorGPU<Dtype> weight_hh,
      TensorGPU<Dtype> bias_ih,
      TensorGPU<Dtype> bias_hh,
      TensorGPU<Dtype> intere_i,
      TensorGPU<Dtype> intere_h) 
  : input_dim_(input_dim), 
    hidden_dim_(hidden_dim),
    weight_ih_(weight_ih),
    weight_hh_(weight_hh),
    bias_ih_(bias_ih),
    bias_hh_(bias_hh),
    intermediate_i(intere_i),
    intermediate_h(intere_h) { }

  virtual ~Cell_GPU() {}
  
  virtual void Forward(
    TensorGPU<Dtype>& input,
    TensorGPU<Dtype>& hidden,
    TensorGPU<Dtype>& output
  ) = 0;

  virtual int hidden_offset_() = 0;


protected:

  int input_dim_, hidden_dim_;

  TensorGPU<Dtype> weight_ih_;
  TensorGPU<Dtype> weight_hh_;
  TensorGPU<Dtype> bias_ih_;
  TensorGPU<Dtype> bias_hh_;

  TensorGPU<Dtype> intermediate_i;
  TensorGPU<Dtype> intermediate_h;

};





template <typename Dtype>
class GRUCell_GPU : public Cell_GPU<Dtype> {
public:
  GRUCell_GPU(
      const int input_dim, const int hidden_dim,
      TensorGPU<Dtype> weight_ih,
      TensorGPU<Dtype> weight_hh,
      TensorGPU<Dtype> bias_ih,
      TensorGPU<Dtype> bias_hh) : 
        Cell_GPU<Dtype>(
          input_dim, hidden_dim, 
          weight_ih, weight_hh,
          bias_ih, bias_hh,
          TensorGPU<Dtype>(3 * hidden_dim),
          TensorGPU<Dtype>(3 * hidden_dim)
        ) {

  }

  virtual ~GRUCell_GPU() {}
  
  virtual void Forward(
    TensorGPU<Dtype>& input,
    TensorGPU<Dtype>& hidden,
    TensorGPU<Dtype>& output
  );


  
  virtual int hidden_offset_() {return this->hidden_dim_;}
  

};

template <typename Dtype>
class LSTMCell_GPU : public Cell_GPU<Dtype> {
public:
  LSTMCell_GPU(
      const int input_dim, const int hidden_dim,
      TensorGPU<Dtype> weight_ih,
      TensorGPU<Dtype> weight_hh,
      TensorGPU<Dtype> bias_ih,
      TensorGPU<Dtype> bias_hh) 
  : Cell_GPU<Dtype>(
      input_dim, hidden_dim, 
      weight_ih, weight_hh,
      bias_ih, bias_hh,
      TensorGPU<Dtype>(4 * hidden_dim),
      TensorGPU<Dtype>(4 * hidden_dim)
    ) { }

  virtual ~LSTMCell_GPU() {}
  
  virtual void Forward(
    TensorGPU<Dtype>& input,
    TensorGPU<Dtype>& hidden,
    TensorGPU<Dtype>& output
  );

  virtual int hidden_offset_() {return 2 * this->hidden_dim_;}


};

template <typename Dtype>
Cell_GPU<Dtype>* cell_factory_gpu_(
    const int input_dim, 
    const int hidden_dim,
    TensorGPU<Dtype> w_ih,
    TensorGPU<Dtype> w_hh,
    TensorGPU<Dtype> b_ih,
    TensorGPU<Dtype> b_hh,
    RNN_CELL_TYPE cell_type) {

  switch (cell_type) {
    case RNN_CELL_TYPE::GRU_CELL: {
      return new hypertea::GRUCell_GPU<Dtype>(input_dim, hidden_dim, w_ih, w_hh, b_ih, b_hh);
    }
    case RNN_CELL_TYPE::LSTM_CELL: {
      return new hypertea::LSTMCell_GPU<Dtype>(input_dim, hidden_dim, w_ih, w_hh, b_ih, b_hh);
    }
    default: {
      std::cout << "Wrong RNN Cell Type!" << std::endl;
      exit(0);
    }
  }

}

template <typename Dtype>
class RNNOp_GPU {

public:
  RNNOp_GPU(
    int input_dim,
    int hidden_dim,
    TensorGPU<Dtype> w_ih,
    TensorGPU<Dtype> w_hh,
    TensorGPU<Dtype> b_ih,
    TensorGPU<Dtype> b_hh,
    RNN_CELL_TYPE cell_type) 
      : input_dim_(input_dim), 
        hidden_dim_(hidden_dim),
        cell_(cell_factory_gpu_<Dtype>(input_dim, hidden_dim, w_ih, w_hh, b_ih, b_hh, cell_type)) {}

  virtual ~RNNOp_GPU() {}

  
  virtual TensorGPU<Dtype> Forward(TensorGPU<Dtype> &input_tensor, TensorGPU<Dtype> &hidden_tensor) = 0;
  

protected:

  int batch_size_ = 1;
  int input_dim_, hidden_dim_;

  std::unique_ptr<Cell_GPU<Dtype>> cell_;


  unsigned int input_offset() {
    return batch_size_ * input_dim_;
  }

  unsigned int output_size() {
    return batch_size_ * hidden_dim_;
  }


  virtual unsigned int output_offset() = 0;
  


};






template <typename Dtype>
class UnidirectionalRNN_GPU : public RNNOp_GPU<Dtype> {

public:
  UnidirectionalRNN_GPU(
    int input_dim,
    int hidden_dim,
    TensorGPU<Dtype> w_ih,
    TensorGPU<Dtype> w_hh,
    TensorGPU<Dtype> b_ih,
    TensorGPU<Dtype> b_hh,
    RNN_CELL_TYPE cell_type) 
      : RNNOp_GPU<Dtype>(input_dim, hidden_dim, w_ih, w_hh, b_ih, b_hh, cell_type) {}

  virtual ~UnidirectionalRNN_GPU() {}


  virtual TensorGPU<Dtype> Forward(TensorGPU<Dtype> &input_tensor, TensorGPU<Dtype> &hidden_tensor);


private:
  virtual unsigned int output_offset() {
    return this->batch_size_ * this->hidden_dim_;
  }


};


template <typename Dtype>
class BidirectionalRNN_GPU : public RNNOp_GPU<Dtype> {

public:
  BidirectionalRNN_GPU(
    int input_dim,
    int hidden_dim,
    TensorGPU<Dtype> w_ih, TensorGPU<Dtype> rw_ih,
    TensorGPU<Dtype> w_hh, TensorGPU<Dtype> rw_hh,
    TensorGPU<Dtype> b_ih, TensorGPU<Dtype> rb_ih,
    TensorGPU<Dtype> b_hh, TensorGPU<Dtype> rb_hh,
    RNN_CELL_TYPE cell_type) 
      : RNNOp_GPU<Dtype>(input_dim, hidden_dim, w_ih, w_hh, b_ih, b_hh, cell_type),
        reverse_cell_(cell_factory_gpu_<Dtype>(input_dim, hidden_dim, rw_ih, rw_hh, rb_ih, rb_hh, cell_type)) { }

  virtual ~BidirectionalRNN_GPU() {}

  virtual TensorGPU<Dtype> Forward(TensorGPU<Dtype> &input_tensor, TensorGPU<Dtype> &hidden_tensor);

private:
  
  virtual unsigned int output_offset() {
    return 2 * this->batch_size_ * this->hidden_dim_;
  }

  std::unique_ptr<Cell_GPU<Dtype>> reverse_cell_;


};


template <typename Dtype>
class StackedRNN_GPU : public GPUFunctor<Dtype> {

public:
  StackedRNN_GPU(
    std::vector<RNNOp_GPU<Dtype>* > rnn_layers) 
      : rnn_layers_(rnn_layers) {}

  ~StackedRNN_GPU()  {
    for (int i = 0; i < rnn_layers_.size(); ++i) {
      delete rnn_layers_[i];
    }
  }

  
  virtual TensorGPU<Dtype> Forward(TensorGPU<Dtype> input_tensor, 
    std::vector<TensorGPU<Dtype> > hidden_tensors);
  
  virtual TensorGPU<Dtype> Forward(TensorGPU<Dtype> &input_tensor) { }

private:

  std::vector<RNNOp_GPU<Dtype>* > rnn_layers_;


};

}  // namespace hypertea

#endif  // HYPERTEA_RNN_OP_HPP_
