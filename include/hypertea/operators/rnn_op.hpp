#ifndef HYPERTEA_RNN_OP_HPP_
#define HYPERTEA_RNN_OP_HPP_

#include <vector>

#include "hypertea/operator.hpp"

namespace hypertea {


enum class RNN_CELL_TYPE {
  LSTM_CELL,
  GRU_CELL
};



template <typename DeviceTensor>
class RNNCell {
public:


  RNNCell(
      const int input_dim, const int hidden_dim,
      const DeviceTensor& weight_ih,
      const DeviceTensor& weight_hh,
      const DeviceTensor& bias_ih,
      const DeviceTensor& bias_hh,
      const DeviceTensor& inter_i,
      const DeviceTensor& inter_h
  ) : input_dim_(input_dim), hidden_dim_(hidden_dim),
      weight_ih_(weight_ih),
      weight_hh_(weight_hh),
      bias_ih_(bias_ih),
      bias_hh_(bias_hh),
      intermediate_i(inter_i),
      intermediate_h(inter_h) { }

  virtual ~RNNCell() {}
  
  virtual void Forward(
    DeviceTensor& input_data,
    DeviceTensor& hidden_data,
    DeviceTensor& output_data
  ) = 0;


  virtual int hidden_offset_() = 0;


protected:

  int input_dim_, hidden_dim_;

  DeviceTensor weight_ih_;
  DeviceTensor weight_hh_;
  DeviceTensor bias_ih_;
  DeviceTensor bias_hh_;

  DeviceTensor intermediate_i;
  DeviceTensor intermediate_h;


};





template <typename DeviceTensor>
class GRUCell : public RNNCell<DeviceTensor> {
public:
  GRUCell(
      const int input_dim, const int hidden_dim,
      const DeviceTensor& weight_ih,
      const DeviceTensor& weight_hh,
      const DeviceTensor& bias_ih,
      const DeviceTensor& bias_hh) : 
        RNNCell<DeviceTensor>(
          input_dim, hidden_dim, 
          weight_ih, weight_hh,
          bias_ih, bias_hh,
          DeviceTensor(3 * hidden_dim),
          DeviceTensor(3 * hidden_dim)
        ) {}

  virtual ~GRUCell() {}
  
  virtual void Forward(
    DeviceTensor& input_data,
    DeviceTensor& hidden_data,
    DeviceTensor& output_data
  );
  
  virtual int hidden_offset_() {return this->hidden_dim_;}
  

};

template <typename DeviceTensor>
class LSTMCell : public RNNCell<DeviceTensor> {
public:
  LSTMCell(
      const int input_dim, const int hidden_dim,
      const DeviceTensor& weight_ih,
      const DeviceTensor& weight_hh,
      const DeviceTensor& bias_ih,
      const DeviceTensor& bias_hh) : 
        RNNCell<DeviceTensor>(
          input_dim, hidden_dim, 
          weight_ih, weight_hh,
          bias_ih, bias_hh,
          DeviceTensor(4 * hidden_dim),
          DeviceTensor(4 * hidden_dim)
        ) { }

  virtual ~LSTMCell() {}
  
  virtual void Forward(
    DeviceTensor& input_data,
    DeviceTensor& hidden_data,
    DeviceTensor& output_data
  );

  virtual int hidden_offset_() {return 2 * this->hidden_dim_;}


};

template <typename DeviceTensor>
RNNCell<DeviceTensor>* cell_factory_(
    const int input_dim, 
    const int hidden_dim,
    const DeviceTensor& w_ih,
    const DeviceTensor& w_hh,
    const DeviceTensor& b_ih,
    const DeviceTensor& b_hh,
    RNN_CELL_TYPE cell_type) {

  switch (cell_type) {
    case RNN_CELL_TYPE::GRU_CELL: {
      return new hypertea::GRUCell<DeviceTensor>(input_dim, hidden_dim, w_ih, w_hh, b_ih, b_hh);
    }
    case RNN_CELL_TYPE::LSTM_CELL: {
      return new hypertea::LSTMCell<DeviceTensor>(input_dim, hidden_dim, w_ih, w_hh, b_ih, b_hh);
    }
    default: {
      std::cout << "Wrong RNN Cell Type!" << std::endl;
      exit(0);
    }
  }

}





template <typename DeviceTensor>
class RNNOp {

public:
  RNNOp(
    int input_dim,
    int hidden_dim,
    const DeviceTensor& w_ih,
    const DeviceTensor& w_hh,
    const DeviceTensor& b_ih,
    const DeviceTensor& b_hh,
    RNN_CELL_TYPE cell_type) 
      : input_dim_(input_dim), 
        hidden_dim_(hidden_dim),
        cell_(cell_factory_<DeviceTensor>(input_dim, hidden_dim, w_ih, w_hh, b_ih, b_hh, cell_type)) {}

  ~RNNOp()  {}

  
  virtual DeviceTensor Forward(DeviceTensor &input_tensor, DeviceTensor &hidden_tensor) = 0;
  

protected:

  int batch_size_ = 1;
  int input_dim_, hidden_dim_;

  std::unique_ptr<RNNCell<DeviceTensor>> cell_;


};






template <typename DeviceTensor>
class UnidirectionalRNN : public RNNOp<DeviceTensor> {

public:
  UnidirectionalRNN(
    int input_dim,
    int hidden_dim,
    const DeviceTensor& w_ih,
    const DeviceTensor& w_hh,
    const DeviceTensor& b_ih,
    const DeviceTensor& b_hh,
    RNN_CELL_TYPE cell_type) 
      : RNNOp<DeviceTensor>(input_dim, hidden_dim, w_ih, w_hh, b_ih, b_hh, cell_type) {}

  ~UnidirectionalRNN() {}


  virtual DeviceTensor Forward(DeviceTensor &input_tensor, DeviceTensor &hidden_tensor);

};


template <typename DeviceTensor>
class BidirectionalRNN : public RNNOp<DeviceTensor> {

public:
  BidirectionalRNN(
    int input_dim,
    int hidden_dim,
    const DeviceTensor& w_ih, const DeviceTensor& rw_ih,
    const DeviceTensor& w_hh, const DeviceTensor& rw_hh,
    const DeviceTensor& b_ih, const DeviceTensor& rb_ih,
    const DeviceTensor& b_hh, const DeviceTensor& rb_hh,
    RNN_CELL_TYPE cell_type) 
      : RNNOp<DeviceTensor>(input_dim, hidden_dim, w_ih, w_hh, b_ih, b_hh, cell_type),
        reverse_cell_(cell_factory_<DeviceTensor>(input_dim, hidden_dim, rw_ih, rw_hh, rb_ih, rb_hh, cell_type)) { }

  ~BidirectionalRNN() {}


  virtual DeviceTensor Forward(DeviceTensor &input_tensor, DeviceTensor &hidden_tensor);

private:
  
  std::unique_ptr<RNNCell<DeviceTensor>> reverse_cell_;


};


template <typename DeviceTensor>
class StackedRNN {

public: 
  StackedRNN(
    std::vector<RNNOp<DeviceTensor>* > rnn_layers) 
      : rnn_layers_(rnn_layers) {}

  ~StackedRNN()  {
    for (int i = 0; i < rnn_layers_.size(); ++i) {
      delete rnn_layers_[i];
    }
  }

  DeviceTensor operator()(DeviceTensor &input, std::vector<DeviceTensor >& hidden_tensors) {
    Forward(input, hidden_tensors);
  }
  DeviceTensor Forward(DeviceTensor &input, std::vector<DeviceTensor >& hidden_tensors);

  

private:

  std::vector<RNNOp<DeviceTensor>* > rnn_layers_;


};



}  // namespace hypertea

#endif  // HYPERTEA_RNN_OP_HPP_
