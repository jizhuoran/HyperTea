#include <vector>

#include "hypertea/operators/rnn_op.hpp"

namespace hypertea {

template <typename DeviceTensor>
void GRUCell<DeviceTensor>::Forward(
    DeviceTensor& input,
    DeviceTensor& hidden,
    DeviceTensor& output
) {

    
    this->intermediate_i.copy_data(this->bias_ih_);
    this->intermediate_h.copy_data(this->bias_hh_);

    inplace_gemv(CblasNoTrans, 3 * this->hidden_dim_,
        this->input_dim_, 1, this->weight_ih_, input, 1, this->intermediate_i);
    inplace_gemv(CblasNoTrans, 3 * this->hidden_dim_,
        this->hidden_dim_, 1, this->weight_hh_, hidden, 1, this->intermediate_h);

    auto igates = this->intermediate_i.chunked_tensors(3);
    auto hgates = this->intermediate_h.chunked_tensors(3);

    inplace_sigmoid(hgates[0] += igates[0]); //reset_gate
    inplace_sigmoid(hgates[1] += igates[1]); //input_gate
    inplace_tanh((hgates[2] *= hgates[0]) += igates[2]); //new_gate

    ((hidden -= hgates[2]) *= hgates[1]) += hgates[2]; //hy = (hx - new_gate) * input_gate + new_gate

    output.copy_data(hidden);

}



template <typename DeviceTensor>
void LSTMCell<DeviceTensor>::Forward(
    DeviceTensor& input,
    DeviceTensor& hidden,
    DeviceTensor& output
) {


    this->intermediate_i.copy_data(this->bias_ih_);
    this->intermediate_h.copy_data(this->bias_hh_);

    inplace_gemv(CblasNoTrans, 4 * this->hidden_dim_,
        this->input_dim_, 1, this->weight_ih_, input, 1, this->intermediate_i);
    inplace_gemv(CblasNoTrans, 4 * this->hidden_dim_,
        this->hidden_dim_, 1, this->weight_hh_, hidden, 1, this->intermediate_h);


    this->intermediate_i += this->intermediate_h;

    auto gates = this->intermediate_i.chunked_tensors(4);
    DeviceTensor& ingate = inplace_sigmoid(gates[0]);
    DeviceTensor& forgetgate = inplace_sigmoid(gates[1]);
    DeviceTensor& cellgate = inplace_tanh(gates[2]);
    DeviceTensor& outgate = inplace_sigmoid(gates[3]);

    auto hiddens = hidden.chunked_tensors(2);
    DeviceTensor& cy = (hiddens[1] *= forgetgate) += (ingate *= cellgate); //cy = cx * forgetgate + (ingate * cellgate)
    DeviceTensor& hy = inplace_tanh(hiddens[0].copy_data(cy)) *= outgate; //hy = cy.tanh() * outgate

    output.copy_data(hy);
}


template <typename DeviceTensor>
DeviceTensor UnidirectionalRNN<DeviceTensor>::Forward(
    DeviceTensor& input_tensor, 
    DeviceTensor& hidden_tensor) {

    int input_length = input_tensor.count() / (this->batch_size_ * this->input_dim_);
    DeviceTensor output_tensor(this->batch_size_ * input_length * this->hidden_dim_);

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

template <typename DeviceTensor>
DeviceTensor BidirectionalRNN<DeviceTensor>::Forward(
    DeviceTensor& input_tensor, 
    DeviceTensor& hidden_tensor) {


    int input_length = input_tensor.count() / (this->batch_size_ * this->input_dim_);
    DeviceTensor output_tensor(2 * this->batch_size_ * input_length * this->hidden_dim_);

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



template <typename DeviceTensor>
DeviceTensor StackedRNN<DeviceTensor>::Forward(
    DeviceTensor &input_tensor, 
    std::vector<DeviceTensor>& hidden_tensors) {

    for (int i = 0; i < rnn_layers_.size(); ++i) {
        input_tensor = rnn_layers_[i]->Forward(input_tensor, hidden_tensors[i]);
        
    }


    return input_tensor;

}



template void GRUCell<TensorCPU<float>>::Forward(TensorCPU<float>& input, TensorCPU<float>& hidden, TensorCPU<float>& output);
template void LSTMCell<TensorCPU<float>>::Forward(TensorCPU<float>& input, TensorCPU<float>& hidden, TensorCPU<float>& output);
template TensorCPU<float> UnidirectionalRNN<TensorCPU<float>>::Forward(TensorCPU<float>& input, TensorCPU<float>& hidden);
template TensorCPU<float> BidirectionalRNN<TensorCPU<float>>::Forward(TensorCPU<float>& input, TensorCPU<float>& hidden);
template TensorCPU<float> StackedRNN<TensorCPU<float>>::Forward(TensorCPU<float>& input, std::vector<TensorCPU<float>> &hidden);



#ifdef USE_OPENCL
template void GRUCell<TensorGPU<float>>::Forward(TensorGPU<float>& input, TensorGPU<float>& hidden, TensorGPU<float>& output);
template void GRUCell<TensorGPU<half>>::Forward(TensorGPU<half>& input, TensorGPU<half>& hidden, TensorGPU<half>& output);

template void LSTMCell<TensorGPU<float>>::Forward(TensorGPU<float>& input, TensorGPU<float>& hidden, TensorGPU<float>& output);
template void LSTMCell<TensorGPU<half>>::Forward(TensorGPU<half>& input, TensorGPU<half>& hidden, TensorGPU<half>& output);

template TensorGPU<float> UnidirectionalRNN<TensorGPU<float>>::Forward(TensorGPU<float>& input, TensorGPU<float>& hidden);
template TensorGPU<half> UnidirectionalRNN<TensorGPU<half>>::Forward(TensorGPU<half>& input, TensorGPU<half>& hidden);

template TensorGPU<float> BidirectionalRNN<TensorGPU<float>>::Forward(TensorGPU<float>& input, TensorGPU<float>& hidden);
template TensorGPU<half> BidirectionalRNN<TensorGPU<half>>::Forward(TensorGPU<half>& input, TensorGPU<half>& hidden);

template TensorGPU<float> StackedRNN<TensorGPU<float>>::Forward(TensorGPU<float>& input, std::vector<TensorGPU<float>> &hidden);
template TensorGPU<half> StackedRNN<TensorGPU<half>>::Forward(TensorGPU<half>& input, std::vector<TensorGPU<half>> &hidden);
#endif //USE_OPENCL
 


}