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
Cell_CPU<Dtype>* cell_factory_(
    const int input_dim, 
    const int hidden_dim,
    Dtype* w_ih,
    Dtype* w_hh,
    Dtype* b_ih,
    Dtype* b_hh,
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
    Dtype* w_ih,
    Dtype* w_hh,
    Dtype* b_ih,
    Dtype* b_hh,
    RNN_CELL_TYPE cell_type) 
      : input_dim_(input_dim), 
        hidden_dim_(hidden_dim),
        cell_(cell_factory_(input_dim, hidden_dim, w_ih, w_hh, b_ih, b_hh, cell_type)) {}

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
    Dtype* w_ih,
    Dtype* w_hh,
    Dtype* b_ih,
    Dtype* b_hh,
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
    Dtype* w_ih, Dtype* rw_ih,
    Dtype* w_hh, Dtype* rw_hh,
    Dtype* b_ih, Dtype* rb_ih,
    Dtype* b_hh, Dtype* rb_hh,
    RNN_CELL_TYPE cell_type) 
      : RNNOp_CPU<Dtype>(input_dim, hidden_dim, w_ih, w_hh, b_ih, b_hh, cell_type),
        reverse_cell_(cell_factory_(input_dim, hidden_dim, rw_ih, rw_hh, rb_ih, rb_hh, cell_type)) { }

  ~BidirectionalRNN_CPU() {}


  virtual TensorCPU<Dtype> Forward(TensorCPU<Dtype> &input_tensor, TensorCPU<Dtype> &hidden_tensor);

private:
  
  std::unique_ptr<Cell_CPU<Dtype>> reverse_cell_;


};


template <typename Dtype>
class StackedRNN : public CPUFunctor<Dtype> {

public:
  StackedRNN(
    std::vector<RNNOp_CPU<Dtype>* > rnn_layers) 
      : rnn_layers_(rnn_layers) {}

  ~StackedRNN()  {
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
      cl_mem weight_ih,
      cl_mem weight_hh,
      cl_mem bias_ih,
      cl_mem bias_hh
  ) : input_dim_(input_dim), hidden_dim_(hidden_dim),
      weight_ih_(weight_ih),
      weight_hh_(weight_hh),
      bias_ih_(bias_ih),
      bias_hh_(bias_hh) { }

  virtual ~Cell_GPU() {}
  
  virtual void Forward(
    cl_mem input_data,
    cl_mem hidden_data,
    cl_mem output_data
  ) = 0;


  virtual int hidden_offset_() = 0;


protected:

  int input_dim_, hidden_dim_;

  cl_mem weight_ih_;
  cl_mem weight_hh_;
  cl_mem bias_ih_;
  cl_mem bias_hh_;

  cl_mem intermediate_i;
  cl_mem intermediate_h;


};





template <typename Dtype>
class GRUCell_GPU : public Cell_GPU<Dtype> {
public:
  GRUCell_GPU(
      const int input_dim, const int hidden_dim,
      cl_mem weight_ih,
      cl_mem weight_hh,
      cl_mem bias_ih,
      cl_mem bias_hh) : 
        Cell_GPU<Dtype>(
          input_dim, hidden_dim, 
          weight_ih, weight_hh,
          bias_ih, bias_hh
        ) {
    
    this->intermediate_i = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 3 * this->hidden_dim_ * sizeof(Dtype), NULL, NULL);
    this->intermediate_h = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 3 * this->hidden_dim_ * sizeof(Dtype), NULL, NULL);


  }

  virtual ~GRUCell_GPU() {
    clReleaseMemObject(this->intermediate_i);
    clReleaseMemObject(this->intermediate_h);
  }
  
  virtual void Forward(
    cl_mem input_data,
    cl_mem hidden_data,
    cl_mem output_data
  );
  
  virtual int hidden_offset_() {return this->hidden_dim_;}
  

};

template <typename Dtype>
class LSTMCell_GPU : public Cell_GPU<Dtype> {
public:
  LSTMCell_GPU(
      const int input_dim, const int hidden_dim,
      cl_mem weight_ih,
      cl_mem weight_hh,
      cl_mem bias_ih,
      cl_mem bias_hh) : 
        Cell_GPU<Dtype>(
          input_dim, hidden_dim, 
          weight_ih, weight_hh,
          bias_ih, bias_hh
        ) {
    
    this->intermediate_i = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 4 * this->hidden_dim_ * sizeof(Dtype), NULL, NULL);
    this->intermediate_h = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 4 * this->hidden_dim_ * sizeof(Dtype), NULL, NULL);

  }

  virtual ~LSTMCell_GPU() {
    clReleaseMemObject(this->intermediate_i);
    clReleaseMemObject(this->intermediate_h);
  }
  
  virtual void Forward(
    cl_mem input_data,
    cl_mem hidden_data,
    cl_mem output_data
  );

  virtual int hidden_offset_() {return 2 * this->hidden_dim_;}


};

template <typename Dtype>
Cell_GPU<Dtype>* cell_factory_(
    const int input_dim, 
    const int hidden_dim,
    cl_mem w_ih,
    cl_mem w_hh,
    cl_mem b_ih,
    cl_mem b_hh,
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
    Dtype* w_ih,
    Dtype* w_hh,
    Dtype* b_ih,
    Dtype* b_hh,
    RNN_CELL_TYPE cell_type) 
      : input_dim_(input_dim), 
        hidden_dim_(hidden_dim),
        cell_(cell_factory_(input_dim, hidden_dim, w_ih, w_hh, b_ih, b_hh, cell_type)) {}

  ~RNNOp_GPU()  {}

  
  virtual TensorGPU<Dtype> Forward(TensorGPU<Dtype> &input_tensor, TensorGPU<Dtype> &hidden_tensor) = 0;
  

protected:

  int batch_size_ = 1;
  int input_dim_, hidden_dim_;

  std::unique_ptr<Cell_GPU<Dtype>> cell_;


};






template <typename Dtype>
class UnidirectionalRNN_GPU : public RNNOp_GPU<Dtype> {

public:
  UnidirectionalRNN_GPU(
    int input_dim,
    int hidden_dim,
    Dtype* w_ih,
    Dtype* w_hh,
    Dtype* b_ih,
    Dtype* b_hh,
    RNN_CELL_TYPE cell_type) 
      : RNNOp_GPU<Dtype>(input_dim, hidden_dim, w_ih, w_hh, b_ih, b_hh, cell_type) {}

  ~UnidirectionalRNN_GPU() {}


  virtual TensorGPU<Dtype> Forward(TensorGPU<Dtype> &input_tensor, TensorGPU<Dtype> &hidden_tensor);

};


template <typename Dtype>
class BidirectionalRNN_GPU : public RNNOp_GPU<Dtype> {

public:
  BidirectionalRNN_GPU(
    int input_dim,
    int hidden_dim,
    Dtype* w_ih, Dtype* rw_ih,
    Dtype* w_hh, Dtype* rw_hh,
    Dtype* b_ih, Dtype* rb_ih,
    Dtype* b_hh, Dtype* rb_hh,
    RNN_CELL_TYPE cell_type) 
      : RNNOp_GPU<Dtype>(input_dim, hidden_dim, w_ih, w_hh, b_ih, b_hh, cell_type),
        reverse_cell_(cell_factory_(input_dim, hidden_dim, rw_ih, rw_hh, rb_ih, rb_hh, cell_type)) { }

  ~BidirectionalRNN_GPU() {}


  virtual TensorGPU<Dtype> Forward(TensorGPU<Dtype> &input_tensor, TensorGPU<Dtype> &hidden_tensor);

private:
  
  std::unique_ptr<Cell_GPU<Dtype>> reverse_cell_;


};


template <typename Dtype>
class StackedRNN : public CPUFunctor<Dtype> {

public:
  StackedRNN(
    std::vector<RNNOp_GPU<Dtype>* > rnn_layers) 
      : rnn_layers_(rnn_layers) {}

  ~StackedRNN()  {
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
