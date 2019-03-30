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







template <typename Dtype>
void GRUCell_GPU<Dtype>::Forward(
    cl_mem input_data,
    cl_mem hidden_data,
    cl_mem output_data
) {

    hypertea_cl_copy<Dtype>(3 * this->hidden_dim_, this->bias_ih_, this->intermediate_i);
    hypertea_cl_copy<Dtype>(3 * this->hidden_dim_, this->bias_hh_, this->intermediate_h);

    hypertea_gpu_gemv<Dtype>(CblasNoTrans, 3 * this->hidden_dim_,
        this->input_dim_, 1, this->weight_ih_, input_data, 1, this->intermediate_i);
    hypertea_gpu_gemv<Dtype>(CblasNoTrans, 3 * this->hidden_dim_,
        this->hidden_dim_, 1, this->weight_hh_, hidden_data, 1, this->intermediate_h);


    hypertea_gpu_add<Dtype>(2 * this->hidden_dim_, this->intermediate_i, this->intermediate_h, this->intermediate_i);
    hypertea_gpu_sigmoid<Dtype>(2 * this->hidden_dim_, this->intermediate_i, this->intermediate_i);



    cl_int ret;

    cl_buffer_region reset_gate_region{0, this->hidden_dim_ * dtype_size_<Dtype>()};
    auto reset_gate = clCreateSubBuffer(this->intermediate_i, CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION, &reset_gate_region, &ret); OPENCL_CHECK(ret);

    // std::cout << "The reference count of inter_i1 is " << reference_count(this->intermediate_i);

    cl_buffer_region input_gate_region{this->hidden_dim_* 1 * dtype_size_<Dtype>(), this->hidden_dim_ * dtype_size_<Dtype>()};
    auto input_gate = clCreateSubBuffer(this->intermediate_i, CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION, &input_gate_region, &ret); OPENCL_CHECK(ret);

    // std::cout << "The reference count of inter_i2 is " << reference_count(this->intermediate_i);


    cl_buffer_region new_gate_region{this->hidden_dim_* 2 * dtype_size_<Dtype>(), this->hidden_dim_ * dtype_size_<Dtype>()};
    auto new_gate_h = clCreateSubBuffer(this->intermediate_h, CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION, &new_gate_region, &ret); OPENCL_CHECK(ret);
    auto new_gate_i = clCreateSubBuffer(this->intermediate_i, CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION, &new_gate_region, &ret); OPENCL_CHECK(ret);

    // std::cout << "The reference count of inter_i3 is " << reference_count(this->intermediate_i);

    // cl_mem reset_gate = this->intermediate_i;
    // cl_mem input_gate = this->intermediate_i + this->hidden_dim_;
    // cl_mem new_gate   = this->intermediate_h + 2 * this->hidden_dim_;


    hypertea_gpu_mul<Dtype>(this->hidden_dim_, reset_gate, new_gate_h, new_gate_h);
    hypertea_gpu_add<Dtype>(this->hidden_dim_, new_gate_i, new_gate_h, new_gate_h);
    hypertea_gpu_tanh<Dtype>(this->hidden_dim_, new_gate_h, new_gate_h);


    hypertea_gpu_sub<Dtype>(this->hidden_dim_, hidden_data, new_gate_h, output_data);
    hypertea_gpu_mul<Dtype>(this->hidden_dim_, input_gate, output_data, output_data);
    hypertea_gpu_add<Dtype>(this->hidden_dim_, new_gate_h, output_data, output_data);

    hypertea_cl_copy<Dtype>(this->hidden_dim_, output_data, hidden_data);

}



template <typename Dtype>
void LSTMCell_GPU<Dtype>::Forward(
    cl_mem input_data,
    cl_mem hidden_data,
    cl_mem output_data
) {

    cl_int ret;

    cl_buffer_region hx_region{0, this->hidden_dim_ * dtype_size_<Dtype>()};
    cl_buffer_region cx_region{this->hidden_dim_ * dtype_size_<Dtype>(), this->hidden_dim_ * dtype_size_<Dtype>()};
    auto hx = clCreateSubBuffer(hidden_data, CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION, &hx_region, &ret); OPENCL_CHECK(ret);
    auto cx = clCreateSubBuffer(hidden_data, CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION, &cx_region, &ret); OPENCL_CHECK(ret);

    hypertea_cl_copy<Dtype>(4 * this->hidden_dim_, this->bias_ih_, this->intermediate_i);
    hypertea_cl_copy<Dtype>(4 * this->hidden_dim_, this->bias_hh_, this->intermediate_h);





    // std::cout << "We are going to do ih " << 4 * this->hidden_dim_ << " and "<< this->input_dim_ << std::endl;
    // std::cout << "We are going to do ih " << cl_mem_count(this->weight_ih_) << " and "<< cl_mem_count(input_data) << std::endl;
    hypertea_gpu_gemv<Dtype>(CblasNoTrans, 4 * this->hidden_dim_,
        this->input_dim_, 1, this->weight_ih_, input_data, 1, this->intermediate_i);

    // std::cout << "We are going to do hh " << 4 * this->hidden_dim_ << " and "<< this->hidden_dim_ << std::endl;
    // std::cout << "We are going to do hh " << cl_mem_count(this->weight_hh_) << " and "<< cl_mem_count(hidden_data) << std::endl;
    hypertea_gpu_gemv<Dtype>(CblasNoTrans, 4 * this->hidden_dim_,
        this->hidden_dim_, 1, this->weight_hh_, hidden_data, 1, this->intermediate_h);

    // Dtype* cpu_data = new Dtype[4 * this->hidden_dim_];
    // OPENCL_CHECK(clEnqueueReadBuffer(OpenCLHandler::Get().commandQueue, (cl_mem)this->intermediate_i, CL_TRUE, 0, sizeof(Dtype) * 4 * this->hidden_dim_, cpu_data, 0, NULL, NULL));
    // for (int i = 0; i < 10; ++i) {
    //     std::cout << cpu_data[i] << " ";
    // }
    // std::cout << " " << std::endl;

    // OPENCL_CHECK(clEnqueueReadBuffer(OpenCLHandler::Get().commandQueue, (cl_mem)this->intermediate_h, CL_TRUE, 0, sizeof(Dtype) * 4 * this->hidden_dim_, cpu_data, 0, NULL, NULL));
    // for (int i = 0; i < 10; ++i) {
    //     std::cout << cpu_data[i] << " ";
    // }
    // std::cout << " " << std::endl;

    hypertea_gpu_add<Dtype>(4 * this->hidden_dim_, this->intermediate_i, this->intermediate_h, this->intermediate_i);
    
    cl_buffer_region ingate_region{0, this->hidden_dim_ * dtype_size_<Dtype>()};
    auto ingate = clCreateSubBuffer(this->intermediate_i, CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION, &ingate_region, &ret); OPENCL_CHECK(ret);
    hypertea_gpu_sigmoid<Dtype>(this->hidden_dim_, ingate, ingate);
    
    cl_buffer_region forgetgate_region{this->hidden_dim_ * dtype_size_<Dtype>(), this->hidden_dim_ * dtype_size_<Dtype>()};
    auto forgetgate = clCreateSubBuffer(this->intermediate_i, CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION, &forgetgate_region, &ret); OPENCL_CHECK(ret);
    hypertea_gpu_sigmoid<Dtype>(this->hidden_dim_, forgetgate, forgetgate);

    cl_buffer_region cellgate_region{this->hidden_dim_*2 * dtype_size_<Dtype>(), this->hidden_dim_ * dtype_size_<Dtype>()};
    auto cellgate = clCreateSubBuffer(this->intermediate_i, CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION, &cellgate_region, &ret); OPENCL_CHECK(ret);
    hypertea_gpu_tanh<Dtype>(this->hidden_dim_, cellgate, cellgate);

    cl_buffer_region outgate_region{this->hidden_dim_*3 * dtype_size_<Dtype>(), this->hidden_dim_ * dtype_size_<Dtype>()};
    auto outgate = clCreateSubBuffer(this->intermediate_i, CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION, &outgate_region, &ret); OPENCL_CHECK(ret);
    hypertea_gpu_sigmoid<Dtype>(this->hidden_dim_, outgate, outgate);
    

    hypertea_gpu_mul<Dtype>(this->hidden_dim_, forgetgate, cx, cx);
    hypertea_gpu_mul<Dtype>(this->hidden_dim_, cellgate, ingate, ingate);
    hypertea_gpu_add<Dtype>(this->hidden_dim_, ingate, cx, cx);

    hypertea_gpu_tanh<Dtype>(this->hidden_dim_, cx, output_data);
    hypertea_gpu_mul<Dtype>(this->hidden_dim_, outgate, output_data, output_data);

    hypertea_cl_copy<Dtype>(this->hidden_dim_, output_data, hx);

}


template <typename Dtype>
TensorGPU<Dtype> UnidirectionalRNN_GPU<Dtype>::Forward(
    TensorGPU<Dtype> &input_tensor, 
    TensorGPU<Dtype> &hidden_tensor) {

    int input_length = input_tensor.count() / (this->batch_size_ * this->input_dim_);
    TensorGPU<Dtype> output_tensor(this->batch_size_ * input_length * this->hidden_dim_);


    // auto input_data = input_tensor.mutable_data();
    auto hidden_data = hidden_tensor.mutable_data();
    // auto output_data = output_tensor.mutable_data();

    cl_int ret;

    for (int i = 0; i < input_length; ++i) {
        
        // std::cout << "This is the " << i << " th input" << std::endl;

        // cl_buffer_region input_region{this->batch_size_ * this->input_dim_ * i * dtype_size_<Dtype>(), this->input_dim_ * dtype_size_<Dtype>()};
        // cl_buffer_region output_region{this->batch_size_ * this->hidden_dim_ * i * dtype_size_<Dtype>(), this->hidden_dim_ * dtype_size_<Dtype>()};
        // auto input_sub_data = clCreateSubBuffer(input_data, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &input_region, &ret); OPENCL_CHECK(ret);
        // auto output_sub_data = clCreateSubBuffer(output_data, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &output_region, &ret); OPENCL_CHECK(ret);

        auto input_sub_data = input_tensor.sub_view(this->input_offset() * i, this->input_offset());
        auto output_sub_data = output_tensor.sub_view(output_offset() * i, output_offset());


        this->cell_->Forward(
            input_sub_data, 
            hidden_data, 
            output_sub_data
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


    cl_mem input_data = input_tensor.mutable_data();
    cl_mem hidden_data = hidden_tensor.mutable_data();
    cl_mem output_data = output_tensor.mutable_data();

    cl_int ret;

    for (int i = 0; i < input_length; ++i) {

        auto input_sub_data = input_tensor.sub_view(this->input_offset() * i, this->input_offset());
        auto output_sub_data = output_tensor.sub_view(output_offset() * i, output_offset());

        this->cell_->Forward(
            input_sub_data, 
            hidden_data, 
            output_sub_data
        );

    }

    cl_buffer_region hidden_region{this->cell_->hidden_offset_() * dtype_size_<Dtype>(), this->cell_->hidden_offset_() * dtype_size_<Dtype>()};
    auto hidden_sub_data = clCreateSubBuffer(hidden_data, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &hidden_region, &ret); OPENCL_CHECK(ret);

    for (int i = 0; i < input_length; ++i) {

        cl_buffer_region input_region{this->batch_size_ * this->input_dim_ * (input_length - 1 - i) * dtype_size_<Dtype>(), this->input_dim_ * dtype_size_<Dtype>()};
        cl_buffer_region output_region{(this->batch_size_ * this->hidden_dim_ + 2 * this->batch_size_ * this->hidden_dim_ * (input_length - 1 - i)) * dtype_size_<Dtype>(), 2 * this->hidden_dim_ * dtype_size_<Dtype>()};
        auto input_sub_data = clCreateSubBuffer(input_data, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &input_region, &ret); OPENCL_CHECK(ret);
        auto output_sub_data = clCreateSubBuffer(output_data, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &output_region, &ret); OPENCL_CHECK(ret);


        this->reverse_cell_->Forward(
            input_sub_data, 
            hidden_sub_data, 
            output_sub_data
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