#include <vector>

#include "hypertea/operators/rnn_op.hpp"

namespace hypertea {





template <typename Dtype>
void GRUCell_CPU<Dtype>::Forward(
    Dtype* input_data,
    Dtype* hidden_data,
    Dtype* output_data
) {

    hypertea_copy<Dtype>(3 * this->hidden_dim_, this->bias_ih_, this->intermediate_i);
    hypertea_copy<Dtype>(3 * this->hidden_dim_, this->bias_hh_, this->intermediate_h);

    hypertea_cpu_gemv<Dtype>(CblasNoTrans, 3 * this->hidden_dim_,
        this->input_dim_, 1, this->weight_ih_, input_data, 1, this->intermediate_i);
    hypertea_cpu_gemv<Dtype>(CblasNoTrans, 3 * this->hidden_dim_,
        this->hidden_dim_, 1, this->weight_hh_, hidden_data, 1, this->intermediate_h);


    hypertea_add<Dtype>(2 * this->hidden_dim_, this->intermediate_i, this->intermediate_h, this->intermediate_i);
    hypertea_sigmoid<Dtype>(2 * this->hidden_dim_, this->intermediate_i, this->intermediate_i);

    Dtype* reset_gate = this->intermediate_i;
    Dtype* input_gate = this->intermediate_i + this->hidden_dim_;
    Dtype* new_gate   = this->intermediate_h + 2 * this->hidden_dim_;


    hypertea_mul<Dtype>(this->hidden_dim_, reset_gate, new_gate, new_gate);
    hypertea_add<Dtype>(this->hidden_dim_, this->intermediate_i + 2*this->hidden_dim_, new_gate, new_gate);
    hypertea_tanh<Dtype>(this->hidden_dim_, new_gate, new_gate);


    hypertea_sub<Dtype>(this->hidden_dim_, hidden_data, new_gate, output_data);
    hypertea_mul<Dtype>(this->hidden_dim_, input_gate, output_data, output_data);
    hypertea_add<Dtype>(this->hidden_dim_, new_gate, output_data, output_data);

    hypertea_copy<Dtype>(this->hidden_dim_, output_data, hidden_data);

}



template <typename Dtype>
void LSTMCell_CPU<Dtype>::Forward(
    Dtype* input_data,
    Dtype* hidden_data,
    Dtype* output_data
) {


    auto hx = hidden_data;
    auto cx = hidden_data + this->hidden_dim_;


    hypertea_copy<Dtype>(4 * this->hidden_dim_, this->bias_ih_, this->intermediate_i);
    hypertea_copy<Dtype>(4 * this->hidden_dim_, this->bias_hh_, this->intermediate_h);

    hypertea_cpu_gemv<Dtype>(CblasNoTrans, 4 * this->hidden_dim_,
        this->input_dim_, 1, this->weight_ih_, input_data, 1, this->intermediate_i);
    hypertea_cpu_gemv<Dtype>(CblasNoTrans, 4 * this->hidden_dim_,
        this->hidden_dim_, 1, this->weight_hh_, hidden_data, 1, this->intermediate_h);


    hypertea_add<Dtype>(4 * this->hidden_dim_, this->intermediate_i, this->intermediate_h, this->intermediate_i);
    auto ingate = this->intermediate_i;
    hypertea_sigmoid<Dtype>(this->hidden_dim_, ingate, ingate);
    
    auto forgetgate = this->intermediate_i + this->hidden_dim_;
    hypertea_sigmoid<Dtype>(this->hidden_dim_, forgetgate, forgetgate);

    auto cellgate = this->intermediate_i + 2 * this->hidden_dim_; //tanh
    hypertea_tanh<Dtype>(this->hidden_dim_, cellgate, cellgate);


    auto outgate = this->intermediate_i + 3 * this->hidden_dim_;
    hypertea_sigmoid<Dtype>(this->hidden_dim_, outgate, outgate);
    

    hypertea_mul<Dtype>(this->hidden_dim_, forgetgate, cx, cx);
    hypertea_mul<Dtype>(this->hidden_dim_, cellgate, ingate, ingate);
    hypertea_add<Dtype>(this->hidden_dim_, ingate, cx, cx);

    hypertea_tanh<Dtype>(this->hidden_dim_, cx, output_data);
    hypertea_mul<Dtype>(this->hidden_dim_, outgate, output_data, output_data);

    hypertea_copy<Dtype>(this->hidden_dim_, output_data, hx);

}


template <typename Dtype>
TensorCPU<Dtype> UnidirectionalRNN_CPU<Dtype>::Forward(
    TensorCPU<Dtype> &input_tensor, 
    TensorCPU<Dtype> &hidden_tensor) {

    int input_length = input_tensor.count() / (this->batch_size_ * this->input_dim_);
    TensorCPU<Dtype> output_tensor(this->batch_size_ * input_length * this->hidden_dim_);


    Dtype* input_data = input_tensor.mutable_data();
    Dtype* hidden_data = hidden_tensor.mutable_data();
    Dtype* output_data = output_tensor.mutable_data();



    for (int i = 0; i < input_length; ++i) {
        this->cell_->Forward(
            input_data, 
            hidden_data, 
            output_data
        );

        input_data += (this->batch_size_ * this->input_dim_);
        // hidden_data = output_data;
        output_data += (this->batch_size_ * this->hidden_dim_);
    }

    return output_tensor;

}

template <typename Dtype>
TensorCPU<Dtype> BidirectionalRNN_CPU<Dtype>::Forward(
    TensorCPU<Dtype> &input_tensor, 
    TensorCPU<Dtype> &hidden_tensor) {


    int input_length = input_tensor.count() / (this->batch_size_ * this->input_dim_);
    TensorCPU<Dtype> output_tensor(2 * this->batch_size_ * input_length * this->hidden_dim_);


    Dtype* input_data = input_tensor.mutable_data();
    Dtype* hidden_data = hidden_tensor.mutable_data();
    Dtype* output_data = output_tensor.mutable_data();

    for (int i = 0; i < input_length; ++i) {
        this->cell_->Forward(
            input_data, 
            hidden_data, 
            output_data
        );

        input_data += (this->batch_size_ * this->input_dim_);
        // hidden_data = output_data;
        output_data += (2 * this->batch_size_ * this->hidden_dim_);
    }

    input_data  -= (this->batch_size_ * this->input_dim_);
    hidden_data = hidden_tensor.mutable_data() + this->cell_->hidden_offset_();
    output_data -= (this->batch_size_ * this->hidden_dim_);

    for (int i = 0; i < input_length; ++i) {
        this->reverse_cell_->Forward(
            input_data, 
            hidden_data, 
            output_data
        );

        input_data -= (this->batch_size_ * this->input_dim_);
        // hidden_data = output_data;
        output_data -= (2 * this->batch_size_ * this->hidden_dim_);
    }

    return output_tensor;


}



template <typename Dtype>
TensorCPU<Dtype> StackedRNN<Dtype>::Forward(
    TensorCPU<Dtype> input_tensor, 
    std::vector<TensorCPU<Dtype> > hidden_tensors) {

    for (int i = 0; i < rnn_layers_.size(); ++i) {
        input_tensor = rnn_layers_[i]->Forward(input_tensor, hidden_tensors[i]);
        
    }


    return input_tensor;

}

INSTANTIATE_CLASS_CPU(StackedRNN);
INSTANTIATE_CLASS_CPU(GRUCell_CPU);
INSTANTIATE_CLASS_CPU(LSTMCell_CPU);
INSTANTIATE_CLASS_CPU(UnidirectionalRNN_CPU);
INSTANTIATE_CLASS_CPU(BidirectionalRNN_CPU);

}