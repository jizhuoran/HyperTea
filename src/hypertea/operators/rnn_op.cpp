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
TensorCPU<Dtype> StackedRNN_CPU<Dtype>::Forward(
    TensorCPU<Dtype> input_tensor, 
    std::vector<TensorCPU<Dtype> > hidden_tensors) {

    for (int i = 0; i < rnn_layers_.size(); ++i) {
        input_tensor = rnn_layers_[i]->Forward(input_tensor, hidden_tensors[i]);
        
    }


    return input_tensor;

}







// template <typename Dtype>
// void GRUCell_GPU<Dtype>::Forward(
//     cl_mem input_data,
//     cl_mem hidden_data,
//     cl_mem output_data
// ) {

//     cl_mem inter_i_data = this->intermediate_i.mutable_data();
//     cl_mem inter_h_data = this->intermediate_h.mutable_data();

//     hypertea_cl_copy<Dtype>(3 * this->hidden_dim_, this->bias_ih_, inter_i_data);
//     hypertea_cl_copy<Dtype>(3 * this->hidden_dim_, this->bias_hh_, inter_h_data);

//     hypertea_gpu_gemv<Dtype>(CblasNoTrans, 3 * this->hidden_dim_,
//         this->input_dim_, 1, this->weight_ih_, input_data, 1, inter_i_data);
//     hypertea_gpu_gemv<Dtype>(CblasNoTrans, 3 * this->hidden_dim_,
//         this->hidden_dim_, 1, this->weight_hh_, hidden_data, 1, inter_h_data);


//     hypertea_gpu_add<Dtype>(2 * this->hidden_dim_, inter_i_data, inter_h_data, inter_i_data);
//     hypertea_gpu_sigmoid<Dtype>(2 * this->hidden_dim_, inter_i_data, inter_i_data);



//     cl_int ret;

//     // cl_buffer_region reset_gate_region{0, this->hidden_dim_ * dtype_size_<Dtype>()};
//     auto reset_gate = this->intermediate_i.sub_view(0, this->hidden_dim_);
//     auto input_gate = this->intermediate_i.sub_view(this->hidden_dim_ , this->hidden_dim_);

//     // clCreateSubBuffer(inter_i_data, CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION, &reset_gate_region, &ret); OPENCL_CHECK(ret);


//     // cl_buffer_region input_gate_region{this->hidden_dim_* 1 * dtype_size_<Dtype>(), this->hidden_dim_ * dtype_size_<Dtype>()};
//     // auto input_gate = clCreateSubBuffer(inter_i_data, CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION, &input_gate_region, &ret); OPENCL_CHECK(ret);


//     // cl_buffer_region new_gate_region{this->hidden_dim_* 2 * dtype_size_<Dtype>(), this->hidden_dim_ * dtype_size_<Dtype>()};
//     // auto new_gate_h = clCreateSubBuffer(inter_h_data, CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION, &new_gate_region, &ret); OPENCL_CHECK(ret);
//     auto new_gate_h = this->intermediate_h.sub_view(this->hidden_dim_ * 2, this->hidden_dim_);//clCreateSubBuffer(inter_i_data, CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION, &new_gate_region, &ret); OPENCL_CHECK(ret);
//     auto new_gate_i = this->intermediate_i.sub_view(this->hidden_dim_ * 2, this->hidden_dim_);//clCreateSubBuffer(inter_i_data, CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION, &new_gate_region, &ret); OPENCL_CHECK(ret);



//     hypertea_gpu_mul<Dtype>(this->hidden_dim_, reset_gate, new_gate_h, new_gate_h);
//     hypertea_gpu_add<Dtype>(this->hidden_dim_, new_gate_i, new_gate_h, new_gate_h);
//     hypertea_gpu_tanh<Dtype>(this->hidden_dim_, new_gate_h, new_gate_h);


//     hypertea_gpu_sub<Dtype>(this->hidden_dim_, hidden_data, new_gate_h, output_data);
//     hypertea_gpu_mul<Dtype>(this->hidden_dim_, input_gate, output_data, output_data);
//     hypertea_gpu_add<Dtype>(this->hidden_dim_, new_gate_h, output_data, output_data);

//     hypertea_cl_copy<Dtype>(this->hidden_dim_, output_data, hidden_data);

// }



// template <typename Dtype>
// void LSTMCell_GPU<Dtype>::Forward(
//     cl_mem input_data,
//     cl_mem hidden_data,
//     cl_mem output_data
// ) {

//     cl_int ret;

//     cl_buffer_region hx_region{0, this->hidden_dim_ * dtype_size_<Dtype>()};
//     cl_buffer_region cx_region{this->hidden_dim_ * dtype_size_<Dtype>(), this->hidden_dim_ * dtype_size_<Dtype>()};
//     auto hx = clCreateSubBuffer(hidden_data, CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION, &hx_region, &ret); OPENCL_CHECK(ret);
//     auto cx = clCreateSubBuffer(hidden_data, CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION, &cx_region, &ret); OPENCL_CHECK(ret);

//     cl_mem inter_i_data = this->intermediate_i.mutable_data();
//     cl_mem inter_h_data = this->intermediate_h.mutable_data();

//     hypertea_cl_copy<Dtype>(4 * this->hidden_dim_, this->bias_ih_, inter_i_data);
//     hypertea_cl_copy<Dtype>(4 * this->hidden_dim_, this->bias_hh_, inter_h_data);


//     hypertea_gpu_gemv<Dtype>(CblasNoTrans, 4 * this->hidden_dim_,
//         this->input_dim_, 1, this->weight_ih_, input_data, 1, inter_i_data);

//     hypertea_gpu_gemv<Dtype>(CblasNoTrans, 4 * this->hidden_dim_,
//         this->hidden_dim_, 1, this->weight_hh_, hidden_data, 1, inter_h_data);


//     this->intermediate_i += this->intermediate_h;

//     // hypertea_gpu_add<Dtype>(4 * this->hidden_dim_, inter_i_data, inter_h_data, inter_i_data);
    
//     auto ingate_tensor = this->intermediate_i.sub_tensor_view(0, this->hidden_dim_).sigmoid();
//     auto ingate = ingate_tensor.mutable_data();
//     // hypertea_gpu_sigmoid<Dtype>(this->hidden_dim_, ingate, ingate);

//     auto forgetgate_tensor = this->intermediate_i.sub_tensor_view(this->hidden_dim_, this->hidden_dim_).sigmoid();
//     auto forgetgate = forgetgate_tensor.mutable_data();
    
//     // hypertea_gpu_sigmoid<Dtype>(this->hidden_dim_, forgetgate, forgetgate);

//     auto cellgate_tensor = this->intermediate_i.sub_tensor_view(this->hidden_dim_ * 2, this->hidden_dim_).tanh();
//     auto cellgate = cellgate_tensor.mutable_data();
    
//     // hypertea_gpu_tanh<Dtype>(this->hidden_dim_, cellgate, cellgate);

//     auto outgate_tensor = this->intermediate_i.sub_tensor_view(this->hidden_dim_ * 3, this->hidden_dim_).sigmoid();
//     auto outgate = outgate_tensor.mutable_data();
    
//     // hypertea_gpu_sigmoid<Dtype>(this->hidden_dim_, outgate, outgate);
    
//     hypertea_gpu_mul<Dtype>(this->hidden_dim_, forgetgate, cx, cx);

//     ingate_tensor *= cellgate_tensor;

//     // hypertea_gpu_mul<Dtype>(this->hidden_dim_, cellgate, ingate, ingate);
//     hypertea_gpu_add<Dtype>(this->hidden_dim_, ingate, cx, cx);

//     hypertea_gpu_tanh<Dtype>(this->hidden_dim_, cx, output_data);
//     hypertea_gpu_mul<Dtype>(this->hidden_dim_, outgate, output_data, output_data);

//     hypertea_cl_copy<Dtype>(this->hidden_dim_, output_data, hx);

// }



template <typename Dtype>
void GRUCell_GPU<Dtype>::Forward(
    TensorGPU<Dtype> & input,
    TensorGPU<Dtype> & hidden,
    TensorGPU<Dtype> & output
) {


    this->intermediate_i.copy_data(this->__bias_ih_);
    this->intermediate_h.copy_data(this->__bias_hh_);

    hypertea_gpu_gemv<Dtype>(CblasNoTrans, 3 * this->hidden_dim_,
        this->input_dim_, 1, this->__weight_ih_, input, 1, this->intermediate_i);
    hypertea_gpu_gemv<Dtype>(CblasNoTrans, 3 * this->hidden_dim_,
        this->hidden_dim_, 1, this->__weight_hh_, hidden, 1, this->intermediate_h);

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

    this->intermediate_i.copy_data(this->__bias_ih_);
    this->intermediate_h.copy_data(this->__bias_hh_);

    hypertea_gpu_gemv<Dtype>(CblasNoTrans, 4 * this->hidden_dim_,
        this->input_dim_, 1, this->__weight_ih_, input, 1, this->intermediate_i);
    hypertea_gpu_gemv<Dtype>(CblasNoTrans, 4 * this->hidden_dim_,
        this->hidden_dim_, 1, this->__weight_hh_, hidden, 1, this->intermediate_h);


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