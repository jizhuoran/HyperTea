#include <vector>

#include "hypertea/operators/rnn_op.hpp"

namespace hypertea {


// template <typename Dtype>
// TensorCPU<Dtype> GRUCell_CPU<Dtype>::Forward(TensorCPU<Dtype> &input_tensor, 
//                                    TensorCPU<Dtype> &hidden_tensor,
//                                    const CellParams<Dtype> & params){

//     // auto chunked_igates = params.linear_ih(input_tensor);
//     // auto chunked_hgates = params.linear_hh(hidden_tensor);

//     // auto reset_gate = hypertea_sigmoid(chunked_igates[0] + chunked_hgates[0]);
//     // auto input_gate = hypertea_sigmoid(chunked_igates[1] + chunked_hgates[1]);

//     // // std::cout << "The pshape is " << reset_gate.shape()[0] << " and " << reset_gate.shape()[1] << std::endl;
//     // // std::cout << "The pshape is " << chunked_hgates[2].shape()[0] << " and " << chunked_hgates[2].shape()[1] << std::endl;
    
//     // auto new_gate = hypertea_tanh(chunked_igates[2] + reset_gate * chunked_hgates[2]);

     
//     // std::cout << "The count is " << chunked_igates[0].count() << " and " << chunked_hgates[0].count() << std::endl;


//     // return new_gate + input_gate * (hidden_tensor - new_gate);

//  }



template <typename Dtype>
void GRUCell_CPU<Dtype>::Forward(
    Dtype* input_data,
    Dtype* hidden_data,
    Dtype* output_data
) {

    hypertea_copy<Dtype>(3 * this->hidden_dim_, this->bias_ih_, intermediate_i);
    hypertea_copy<Dtype>(3 * this->hidden_dim_, this->bias_hh_, intermediate_h);

    hypertea_cpu_gemv<Dtype>(CblasNoTrans, 3 * this->hidden_dim_,
        this->input_dim_, 1, this->weight_ih_, input_data, 1, intermediate_i);
    hypertea_cpu_gemv<Dtype>(CblasNoTrans, 3 * this->hidden_dim_,
        this->hidden_dim_, 1, this->weight_hh_, hidden_data, 1, intermediate_h);


    hypertea_add<Dtype>(2 * this->hidden_dim_, intermediate_i, intermediate_h, intermediate_i);
    hypertea_sigmoid<Dtype>(2 * this->hidden_dim_, intermediate_i, intermediate_i);

    Dtype* reset_gate = intermediate_i;
    Dtype* input_gate = intermediate_i + this->hidden_dim_;
    Dtype* new_gate   = intermediate_h + 2 * this->hidden_dim_;


    hypertea_mul<Dtype>(this->hidden_dim_, reset_gate, new_gate, new_gate);
    hypertea_add<Dtype>(this->hidden_dim_, intermediate_i + 2*this->hidden_dim_, new_gate, new_gate);
    hypertea_tanh<Dtype>(this->hidden_dim_, new_gate, new_gate);


    hypertea_sub<Dtype>(this->hidden_dim_, hidden_data, new_gate, output_data);
    hypertea_mul<Dtype>(this->hidden_dim_, input_gate, output_data, output_data);
    hypertea_add<Dtype>(this->hidden_dim_, new_gate, output_data, output_data);

}


template <typename Dtype>
TensorCPU<Dtype> GRUOp_CPU<Dtype>::Forward(
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
        hidden_data = output_data;
        output_data += (this->batch_size_ * this->hidden_dim_);
    }

    return output_tensor;

}





INSTANTIATE_CLASS_CPU(GRUCell_CPU);
INSTANTIATE_CLASS_CPU(GRUOp_CPU);

}