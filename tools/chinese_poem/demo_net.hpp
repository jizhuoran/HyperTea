#include "hypertea/hypertea.hpp"

namespace hypertea {

template <typename DeviceTensor>
class AttenNet {

public:

    AttenNet(const std::string &param_file) { 

        compile_opencl_kernels(" ", " ");
        
        load_weight_to_tensor(param_file, param);

    }

    
    
    void inference( std::vector<int> &data_from_user, std::vector<float> &data_to_user) {
        
        // TensorCPU<float> data(data_from_user);
        auto hidden = std::vector<DeviceTensor>{DeviceTensor(128, 0)};


        auto embeds = embedding(data_from_user);

        auto encoder_inputs = embeds.sub_view(128, 128 * 24);
        auto decoder_inputs = embeds.sub_view(0, 128 * 24);



        auto encoder_out = encoder.Forward(encoder_inputs, hidden);
        auto decoder_out = decoder.Forward(decoder_inputs, hidden);

        auto encoder_outs = encoder_out.chunked_tensors(24);

        encoder_out = concate(std::vector<DeviceTensor*> { &encoder_outs[0], &encoder_outs[6], &encoder_outs[12], &encoder_outs[18]});

        auto attn_mid = attn_mul(encoder_out);


        auto attn_weights = outplace_gemm(
            CblasNoTrans, CblasTrans, 
            24, 4, 128,
            1.0,
            decoder_out, 
            attn_mid,
            0.0
        );

        // attn_weights = inplace_softmax(attn_weights, 24)
        



        auto attn_applied = outplace_gemm(
            CblasNoTrans, CblasNoTrans, 
            24, 128, 4,
            1.0,
            attn_weights, 
            encoder_out,
            0.0
        );

        // attn_applied = torch.bmm(attn_weights, encoder_outputs)

        auto output = hconcate(std::vector<DeviceTensor*> {&attn_applied, &decoder_out}, 24);


        // std::cout << " " << std::endl;

        // auto temp = model(data);

        // hypertea_copy(data_to_user.size(), temp.data(), data_to_user.data());

    }


private:
    
    
    DeviceTensor param = DeviceTensor(2766703);

     DeviceTensor embedding_weight = param.sub_view(0, 636800);
     DeviceTensor encoder_weight_ih_l0 = param.sub_view(636800, 49152);
     DeviceTensor encoder_weight_hh_l0 = param.sub_view(685952, 49152);
     DeviceTensor encoder_bias_ih_l0 = param.sub_view(735104, 384);
     DeviceTensor encoder_bias_hh_l0 = param.sub_view(735488, 384);

     DeviceTensor decoder_weight_ih_l0 = param.sub_view(1372672, 49152);
     DeviceTensor decoder_weight_hh_l0 = param.sub_view(1421824, 49152);
     DeviceTensor decoder_bias_ih_l0 = param.sub_view(1470976, 384);
     DeviceTensor decoder_bias_hh_l0 = param.sub_view(1471360, 384);
     DeviceTensor attn_mul_weight = param.sub_view(1471744, 16384);
     DeviceTensor out_weight = param.sub_view(1488128, 1273600);
     DeviceTensor out_bias = param.sub_view(2761728, 4975);

    EmbeddingOp<DeviceTensor> embedding = EmbeddingOp<DeviceTensor> ( &embedding_weight, 128 );
    StackedRNN<DeviceTensor> encoder = StackedRNN<DeviceTensor> (
            std::vector<hypertea::RNNOp<DeviceTensor>* > {
                new hypertea::UnidirectionalRNN<DeviceTensor> ( 128, 128, encoder_weight_ih_l0, encoder_weight_hh_l0, encoder_bias_ih_l0, encoder_bias_hh_l0, hypertea::RNN_CELL_TYPE::GRU_CELL )
            }
            );

    StackedRNN<DeviceTensor> decoder = StackedRNN<DeviceTensor> (
            std::vector<hypertea::RNNOp<DeviceTensor>* > {
                new hypertea::UnidirectionalRNN<DeviceTensor> ( 128, 128, decoder_weight_ih_l0, decoder_weight_hh_l0, decoder_bias_ih_l0, decoder_bias_hh_l0, hypertea::RNN_CELL_TYPE::GRU_CELL )
            }
            );
    LinearOp<DeviceTensor> attn_mul = LinearOp<DeviceTensor> ( &attn_mul_weight, nullptr, 128, 128 );
    LinearOp<DeviceTensor> out = LinearOp<DeviceTensor> ( &out_weight, &out_bias, 256, 4975 );


};


} //namespace hypertea