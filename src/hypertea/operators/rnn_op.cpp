#include <vector>

#include "hypertea/operators/rnn_op.hpp"

namespace hypertea {





template <typename Dtype>
void GRUCell_CPU<Dtype>::Forward(
    TensorCPU<Dtype>& input,
    TensorCPU<Dtype>& hidden,
    TensorCPU<Dtype>& output
) {

    
    this->intermediate_i.copy_data(this->bias_ih_);
    this->intermediate_h.copy_data(this->bias_hh_);

    inplace_cpu_gemv<Dtype>(CblasNoTrans, 3 * this->hidden_dim_,
        this->input_dim_, 1, this->weight_ih_, input, 1, this->intermediate_i);
    inplace_cpu_gemv<Dtype>(CblasNoTrans, 3 * this->hidden_dim_,
        this->hidden_dim_, 1, this->weight_hh_, hidden, 1, this->intermediate_h);

    auto igates = this->intermediate_i.chunked_tensors(3);
    auto hgates = this->intermediate_h.chunked_tensors(3);

    inplace_cpu_sigmoid(hgates[0] += igates[0]); //reset_gate
    inplace_cpu_sigmoid(hgates[1] += igates[1]); //input_gate
    inplace_cpu_tanh((hgates[2] *= hgates[0]) += igates[2]); //new_gate

    ((hidden -= hgates[2]) *= hgates[1]) += hgates[2]; //hy = (hx - new_gate) * input_gate + new_gate

    output.copy_data(hidden);

}



template <typename Dtype>
void LSTMCell_CPU<Dtype>::Forward(
    TensorCPU<Dtype>& input,
    TensorCPU<Dtype>& hidden,
    TensorCPU<Dtype>& output
) {


    this->intermediate_i.copy_data(this->bias_ih_);
    this->intermediate_h.copy_data(this->bias_hh_);

    inplace_cpu_gemv<Dtype>(CblasNoTrans, 4 * this->hidden_dim_,
        this->input_dim_, 1, this->weight_ih_, input, 1, this->intermediate_i);
    inplace_cpu_gemv<Dtype>(CblasNoTrans, 4 * this->hidden_dim_,
        this->hidden_dim_, 1, this->weight_hh_, hidden, 1, this->intermediate_h);


    this->intermediate_i += this->intermediate_h;

    auto gates = this->intermediate_i.chunked_tensors(4);
    TensorCPU<Dtype>& ingate = gates[0].sigmoid();
    TensorCPU<Dtype>& forgetgate = gates[1].sigmoid();
    TensorCPU<Dtype>& cellgate = gates[2].tanh();
    TensorCPU<Dtype>& outgate = gates[3].sigmoid();

    auto hiddens = hidden.chunked_tensors(2);
    TensorCPU<Dtype>& cy = (hiddens[1] *= forgetgate) += (ingate *= cellgate); //cy = cx * forgetgate + (ingate * cellgate)
    TensorCPU<Dtype>& hy = inplace_cpu_tanh(hiddens[0].copy_data(cy)) *= outgate; //hy = cy.tanh() * outgate

    output.copy_data(hy);
}


template <typename Dtype>
TensorCPU<Dtype> UnidirectionalRNN_CPU<Dtype>::Forward(
    TensorCPU<Dtype> &input_tensor, 
    TensorCPU<Dtype> &hidden_tensor) {

    int input_length = input_tensor.count() / (this->batch_size_ * this->input_dim_);
    TensorCPU<Dtype> output_tensor(this->batch_size_ * input_length * this->hidden_dim_);

    auto input_tensors = input_tensor.chunked_tensors(input_length);
    auto output_tensors = output_tensor.chunked_tensors(input_length);

    for (int i = 0; i < input_length; ++i) {
        
        this->cell_->Forward(
            input_tensors[i], 
            hidden_tensor, 
            output_tensors[i]
        );
    }

    return output_tensor;

}

template <typename Dtype>
TensorCPU<Dtype> BidirectionalRNN_CPU<Dtype>::Forward(
    TensorCPU<Dtype> &input_tensor, 
    TensorCPU<Dtype> &hidden_tensor) {


    int input_length = input_tensor.count() / (this->batch_size_ * this->input_dim_);
    TensorCPU<Dtype> output_tensor(2 * this->batch_size_ * input_length * this->hidden_dim_);

    auto input_tensors = input_tensor.chunked_tensors(input_length);
    auto hidden_tensors = hidden_tensor.chunked_tensors(2);
    auto output_tensors = output_tensor.chunked_tensors(input_length * 2);


    for (int i = 0; i < input_length; ++i) {

        this->cell_->Forward(
            input_tensors[i], 
            hidden_tensors[0], 
            output_tensors[2*i]
        );
    }

    for (int i = input_length - 1; i >= 0; --i) {

        this->reverse_cell_->Forward(
            input_tensors[i], 
            hidden_tensors[1], 
            output_tensors[2*i + 1]
        );
    }

    return output_tensor;


}



template <typename Dtype>
TensorCPU<Dtype> StackedRNN_CPU<Dtype>::Forward(
    TensorCPU<Dtype> input_tensor, 
    std::vector<TensorCPU<Dtype> > hidden_tensors) {

    for (int i = 0; i < rnn_layers_.size(); ++i) {
        input_tensor = rnn_layers_[i]->Forward(input_tensor, hidden_tensors[i]);
        
    }


    return input_tensor;

}



template <typename Dtype>
void GRUCell_GPU<Dtype>::Forward(
    TensorGPU<Dtype> & input,
    TensorGPU<Dtype> & hidden,
    TensorGPU<Dtype> & output
) {


    this->intermediate_i.copy_data(this->bias_ih_);
    this->intermediate_h.copy_data(this->bias_hh_);

    hypertea_gpu_gemv<Dtype>(CblasNoTrans, 3 * this->hidden_dim_,
        this->input_dim_, 1, this->weight_ih_, input, 1, this->intermediate_i);
    hypertea_gpu_gemv<Dtype>(CblasNoTrans, 3 * this->hidden_dim_,
        this->hidden_dim_, 1, this->weight_hh_, hidden, 1, this->intermediate_h);

    auto igates = this->intermediate_i.chunked_tensors(3);
    auto hgates = this->intermediate_h.chunked_tensors(3);

    inplace_gpu_sigmoid(hgates[0] += igates[0]); //reset_gate
    inplace_gpu_sigmoid(hgates[1] += igates[1]); //input_gate
    inplace_gpu_tanh((hgates[2] *= hgates[0]) += igates[2]); //new_gate

    ((hidden -= hgates[2]) *= hgates[1]) += hgates[2]; //hy = (hx - new_gate) * input_gate + new_gate

    output.copy_data(hidden);

}


template <typename Dtype>
void LSTMCell_GPU<Dtype>::Forward(
    TensorGPU<Dtype> & input,
    TensorGPU<Dtype> & hidden,
    TensorGPU<Dtype> & output
) {

    this->intermediate_i.copy_data(this->bias_ih_);
    this->intermediate_h.copy_data(this->bias_hh_);

    hypertea_gpu_gemv<Dtype>(CblasNoTrans, 4 * this->hidden_dim_,
        this->input_dim_, 1, this->weight_ih_, input, 1, this->intermediate_i);
    hypertea_gpu_gemv<Dtype>(CblasNoTrans, 4 * this->hidden_dim_,
        this->hidden_dim_, 1, this->weight_hh_, hidden, 1, this->intermediate_h);


    this->intermediate_i += this->intermediate_h;

    auto gates = this->intermediate_i.chunked_tensors(4);
    TensorGPU<Dtype>& ingate = gates[0].sigmoid();
    TensorGPU<Dtype>& forgetgate = gates[1].sigmoid();
    TensorGPU<Dtype>& cellgate = gates[2].tanh();
    TensorGPU<Dtype>& outgate = gates[3].sigmoid();

    auto hiddens = hidden.chunked_tensors(2);
    TensorGPU<Dtype>& cy = (hiddens[1] *= forgetgate) += (ingate *= cellgate); //cy = cx * forgetgate + (ingate * cellgate)
    TensorGPU<Dtype>& hy = inplace_gpu_tanh(hiddens[0].copy_data(cy)) *= outgate; //hy = cy.tanh() * outgate

    output.copy_data(hy);

}

template <typename Dtype>
TensorGPU<Dtype> UnidirectionalRNN_GPU<Dtype>::Forward(
    TensorGPU<Dtype> &input_tensor, 
    TensorGPU<Dtype> &hidden_tensor) {

    int input_length = input_tensor.count() / (this->batch_size_ * this->input_dim_);
    TensorGPU<Dtype> output_tensor(this->batch_size_ * input_length * this->hidden_dim_);

    auto input_tensors = input_tensor.chunked_tensors(input_length);
    auto output_tensors = output_tensor.chunked_tensors(input_length);

    for (int i = 0; i < input_length; ++i) {
        
        this->cell_->Forward(
            input_tensors[i], 
            hidden_tensor, 
            output_tensors[i]
        );
    }

    return output_tensor;

}


template <typename Dtype>
TensorGPU<Dtype> BidirectionalRNN_GPU<Dtype>::Forward(
    TensorGPU<Dtype> &input_tensor, 
    TensorGPU<Dtype> &hidden_tensor) {


    int input_length = input_tensor.count() / (this->batch_size_ * this->input_dim_);
    TensorGPU<Dtype> output_tensor(2 * this->batch_size_ * input_length * this->hidden_dim_);

    auto input_tensors = input_tensor.chunked_tensors(input_length);
    auto hidden_tensors = hidden_tensor.chunked_tensors(2);
    auto output_tensors = output_tensor.chunked_tensors(input_length * 2);


    for (int i = 0; i < input_length; ++i) {

        this->cell_->Forward(
            input_tensors[i], 
            hidden_tensors[0], 
            output_tensors[2*i]
        );
    }

    for (int i = input_length - 1; i >= 0; --i) {

        this->reverse_cell_->Forward(
            input_tensors[i], 
            hidden_tensors[1], 
            output_tensors[2*i + 1]
        );
    }

    return output_tensor;


}



template <typename Dtype>
TensorGPU<Dtype> StackedRNN_GPU<Dtype>::Forward(
    TensorGPU<Dtype> input_tensor, 
    std::vector<TensorGPU<Dtype> > hidden_tensors) {

    for (int i = 0; i < rnn_layers_.size(); ++i) {
        input_tensor = rnn_layers_[i]->Forward(input_tensor, hidden_tensors[i]);
    }

    return input_tensor;

}


INSTANTIATE_CLASS_CPU(GRUCell_CPU);
INSTANTIATE_CLASS_CPU(LSTMCell_CPU);
INSTANTIATE_CLASS_CPU(UnidirectionalRNN_CPU);
INSTANTIATE_CLASS_CPU(BidirectionalRNN_CPU);
INSTANTIATE_CLASS_CPU(StackedRNN_CPU);


INSTANTIATE_CLASS_GPU(GRUCell_GPU);
INSTANTIATE_CLASS_GPU(LSTMCell_GPU);
INSTANTIATE_CLASS_GPU(UnidirectionalRNN_GPU);
INSTANTIATE_CLASS_GPU(BidirectionalRNN_GPU);
INSTANTIATE_CLASS_GPU(StackedRNN_GPU);

}