#include <vector>

#include "hypertea/operators/rnn_op.hpp"

namespace hypertea {


template <typename Dtype>
TensorCPU<Dtype> GRUCell_CPU<Dtype>::Forward(TensorCPU<Dtype> &input_tensor, 
                                   TensorCPU<Dtype> &hidden_tensor,
                                   const CellParams<Dtype> & params){

    auto chunked_igates = params.linear_ih(input_tensor);
    auto chunked_hgates = params.linear_hh(hidden_tensor);

    auto reset_gate = hypertea_sigmoid(chunked_igates[0] + chunked_hgates[0]);
    auto input_gate = hypertea_sigmoid(chunked_igates[1] + chunked_hgates[1]);

    std::cout << "The pshape is " << reset_gate.shape()[0] << " and " << reset_gate.shape()[1] << std::endl;
    std::cout << "The pshape is " << chunked_hgates[2].shape()[0] << " and " << chunked_hgates[2].shape()[1] << std::endl;
    
    auto new_gate = hypertea_tanh(chunked_igates[2] + reset_gate * chunked_hgates[2]);

    


    return new_gate + input_gate * (hidden_tensor - new_gate);

 }



INSTANTIATE_CLASS_CPU(GRUCell_CPU);

}