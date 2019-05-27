#include "hypertea/hypertea.hpp"
#include "kernels/conv_kernel.cl"

namespace hypertea {

template <typename DeviceTensor>
class new_net {

public:

    new_net(const std::string &param_file) { 

        compile_opencl_kernels(conv_opencl_funcs, " ", true);
        
        // load_weight_to_tensor(param_file, param);

        auto all_weights = load_weights<float>(param_file, 1821315);

        conv1_bias.copy_from_ptr(all_weights + 0);
        conv1_weight.copy_from_ptr(all_weights + 32);
        bn1_weight.copy_from_ptr(all_weights + 7808);
        bn1_bias.copy_from_ptr(all_weights + 7840);
        conv2_bias.copy_from_ptr(all_weights + 7872);
        conv2_weight.copy_from_ptr(all_weights + 7936);
        bn2_weight.copy_from_ptr(all_weights + 40704);
        bn2_bias.copy_from_ptr(all_weights + 40768);
        conv3_bias.copy_from_ptr(all_weights + 40832);
        conv3_weight.copy_from_ptr(all_weights + 40960);
        bn3_weight.copy_from_ptr(all_weights + 172032);
        bn3_bias.copy_from_ptr(all_weights + 172160);
        res1_conv1_weight.copy_from_ptr(all_weights + 172288);
        res1_bn1_weight.copy_from_ptr(all_weights + 319744);
        res1_bn1_bias.copy_from_ptr(all_weights + 319872);
        res1_conv2_weight.copy_from_ptr(all_weights + 320000);
        res1_bn2_weight.copy_from_ptr(all_weights + 467456);
        res1_bn2_bias.copy_from_ptr(all_weights + 467584);
        res2_conv1_weight.copy_from_ptr(all_weights + 467712);
        res2_bn1_weight.copy_from_ptr(all_weights + 615168);
        res2_bn1_bias.copy_from_ptr(all_weights + 615296);
        res2_conv2_weight.copy_from_ptr(all_weights + 615424);
        res2_bn2_weight.copy_from_ptr(all_weights + 762880);
        res2_bn2_bias.copy_from_ptr(all_weights + 763008);
        res3_conv1_weight.copy_from_ptr(all_weights + 763136);
        res3_bn1_weight.copy_from_ptr(all_weights + 910592);
        res3_bn1_bias.copy_from_ptr(all_weights + 910720);
        res3_conv2_weight.copy_from_ptr(all_weights + 910848);
        res3_bn2_weight.copy_from_ptr(all_weights + 1058304);
        res3_bn2_bias.copy_from_ptr(all_weights + 1058432);
        res4_conv1_weight.copy_from_ptr(all_weights + 1058560);
        res4_bn1_weight.copy_from_ptr(all_weights + 1206016);
        res4_bn1_bias.copy_from_ptr(all_weights + 1206144);
        res4_conv2_weight.copy_from_ptr(all_weights + 1206272);
        res4_bn2_weight.copy_from_ptr(all_weights + 1353728);
        res4_bn2_bias.copy_from_ptr(all_weights + 1353856);
        res5_conv1_weight.copy_from_ptr(all_weights + 1353984);
        res5_bn1_weight.copy_from_ptr(all_weights + 1501440);
        res5_bn1_bias.copy_from_ptr(all_weights + 1501568);
        res5_conv2_weight.copy_from_ptr(all_weights + 1501696);
        res5_bn2_weight.copy_from_ptr(all_weights + 1649152);
        res5_bn2_bias.copy_from_ptr(all_weights + 1649280);
        deconv1_bias.copy_from_ptr(all_weights + 1649408);
        deconv1_weight.copy_from_ptr(all_weights + 1649472);
        de_bn1_weight.copy_from_ptr(all_weights + 1780544);
        de_bn1_bias.copy_from_ptr(all_weights + 1780608);
        deconv2_bias.copy_from_ptr(all_weights + 1780672);
        deconv2_weight.copy_from_ptr(all_weights + 1780704);
        de_bn2_weight.copy_from_ptr(all_weights + 1813472);
        de_bn2_bias.copy_from_ptr(all_weights + 1813504);
        deconv3_bias.copy_from_ptr(all_weights + 1813536);
        deconv3_weight.copy_from_ptr(all_weights + 1813539);
    
        free(all_weights);
    
    }

    
    
    void inference( std::vector<int16> &data_from_user, std::vector<int16> &data_to_user) {
        
        auto data = DeviceTensor(data_from_user);

        auto temp = bn1(inplace_elu(conv1(data)));

        temp = bn2(inplace_elu(conv2(temp)));
        temp = bn3(inplace_elu(conv3(temp)));


        temp += res1_bn2(res1_conv2(inplace_relu(res1_bn1(res1_conv1(temp)))));
        temp += res2_bn2(res2_conv2(inplace_relu(res2_bn1(res2_conv1(temp)))));
        temp += res3_bn2(res3_conv2(inplace_relu(res3_bn1(res3_conv1(temp)))));
        temp += res4_bn2(res4_conv2(inplace_relu(res4_bn1(res4_conv1(temp)))));
        temp += res5_bn2(res5_conv2(inplace_relu(res5_bn1(res5_conv1(temp)))));
        

        temp = de_bn1(inplace_elu(deconv1(temp)));
        temp = de_bn2(inplace_elu(deconv2(temp)));
        temp = inplace_tanh(deconv3(temp));

        temp = (temp + 1) * 127.5;

        temp.copy_to_ptr((void*)data_to_user.data());
    }


private:
    


    

    DeviceTensor conv1_bias = DeviceTensor(32);
    DeviceTensor conv1_weight = DeviceTensor(7776);
    DeviceTensor bn1_weight = DeviceTensor(32);
    DeviceTensor bn1_bias = DeviceTensor(32);
    DeviceTensor conv2_bias = DeviceTensor(64);
    DeviceTensor conv2_weight = DeviceTensor(32768);
    DeviceTensor bn2_weight = DeviceTensor(64);
    DeviceTensor bn2_bias = DeviceTensor(64);
    DeviceTensor conv3_bias = DeviceTensor(128);
    DeviceTensor conv3_weight = DeviceTensor(131072);
    DeviceTensor bn3_weight = DeviceTensor(128);
    DeviceTensor bn3_bias = DeviceTensor(128);
    DeviceTensor res1_conv1_weight = DeviceTensor(147456);
    DeviceTensor res1_bn1_weight = DeviceTensor(128);
    DeviceTensor res1_bn1_bias = DeviceTensor(128);
    DeviceTensor res1_conv2_weight = DeviceTensor(147456);
    DeviceTensor res1_bn2_weight = DeviceTensor(128);
    DeviceTensor res1_bn2_bias = DeviceTensor(128);
    DeviceTensor res2_conv1_weight = DeviceTensor(147456);
    DeviceTensor res2_bn1_weight = DeviceTensor(128);
    DeviceTensor res2_bn1_bias = DeviceTensor(128);
    DeviceTensor res2_conv2_weight = DeviceTensor(147456);
    DeviceTensor res2_bn2_weight = DeviceTensor(128);
    DeviceTensor res2_bn2_bias = DeviceTensor(128);
    DeviceTensor res3_conv1_weight = DeviceTensor(147456);
    DeviceTensor res3_bn1_weight = DeviceTensor(128);
    DeviceTensor res3_bn1_bias = DeviceTensor(128);
    DeviceTensor res3_conv2_weight = DeviceTensor(147456);
    DeviceTensor res3_bn2_weight = DeviceTensor(128);
    DeviceTensor res3_bn2_bias = DeviceTensor(128);
    DeviceTensor res4_conv1_weight = DeviceTensor(147456);
    DeviceTensor res4_bn1_weight = DeviceTensor(128);
    DeviceTensor res4_bn1_bias = DeviceTensor(128);
    DeviceTensor res4_conv2_weight = DeviceTensor(147456);
    DeviceTensor res4_bn2_weight = DeviceTensor(128);
    DeviceTensor res4_bn2_bias = DeviceTensor(128);
    DeviceTensor res5_conv1_weight = DeviceTensor(147456);
    DeviceTensor res5_bn1_weight = DeviceTensor(128);
    DeviceTensor res5_bn1_bias = DeviceTensor(128);
    DeviceTensor res5_conv2_weight = DeviceTensor(147456);
    DeviceTensor res5_bn2_weight = DeviceTensor(128);
    DeviceTensor res5_bn2_bias = DeviceTensor(128);
    DeviceTensor deconv1_bias = DeviceTensor(64);
    DeviceTensor deconv1_weight = DeviceTensor(131072);
    DeviceTensor de_bn1_weight = DeviceTensor(64);
    DeviceTensor de_bn1_bias = DeviceTensor(64);
    DeviceTensor deconv2_bias = DeviceTensor(32);
    DeviceTensor deconv2_weight = DeviceTensor(32768);
    DeviceTensor de_bn2_weight = DeviceTensor(32);
    DeviceTensor de_bn2_bias = DeviceTensor(32);
    DeviceTensor deconv3_bias = DeviceTensor(3);
    DeviceTensor deconv3_weight = DeviceTensor(7776);


    LibDNNConvOp<DeviceTensor> conv1 = LibDNNConvOp<DeviceTensor> ("conv1_forward", 8388608, &conv1_weight, &conv1_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {32768,8,1});
    ELUOp<DeviceTensor> elu1 = ELUOp<DeviceTensor> ( 1, NOT_IN_PLACE );
    BatchNormOp<DeviceTensor> bn1 = BatchNormOp<DeviceTensor> (32, 262144, 1e-05, nullptr, nullptr, &bn1_weight, &bn1_bias);
    LibDNNConvOp<DeviceTensor> conv2 = LibDNNConvOp<DeviceTensor> ("conv2_forward", 4194304, &conv2_weight, &conv2_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {8192,16,1});
    ELUOp<DeviceTensor> elu2 = ELUOp<DeviceTensor> ( 1, NOT_IN_PLACE );
    BatchNormOp<DeviceTensor> bn2 = BatchNormOp<DeviceTensor> (64, 65536, 1e-05, nullptr, nullptr, &bn2_weight, &bn2_bias);
    LibDNNConvOp<DeviceTensor> conv3 = LibDNNConvOp<DeviceTensor> ("conv3_forward", 2097152, &conv3_weight, &conv3_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {2048,32,1});
    ELUOp<DeviceTensor> elu3 = ELUOp<DeviceTensor> ( 1, NOT_IN_PLACE );
    BatchNormOp<DeviceTensor> bn3 = BatchNormOp<DeviceTensor> (128, 16384, 1e-05, nullptr, nullptr, &bn3_weight, &bn3_bias);
    LibDNNConvOp<DeviceTensor> res1_conv1 = LibDNNConvOp<DeviceTensor> ("res1_conv1_forward", 2097152, &res1_conv1_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {2048,32,1});
    BatchNormOp<DeviceTensor> res1_bn1 = BatchNormOp<DeviceTensor> (128, 16384, 1e-05, nullptr, nullptr, &res1_bn1_weight, &res1_bn1_bias);
    ReLUOp<DeviceTensor> res1_relu1 = ReLUOp<DeviceTensor> ( 0, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> res1_conv2 = LibDNNConvOp<DeviceTensor> ("res1_conv2_forward", 2097152, &res1_conv2_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {2048,32,1});
    BatchNormOp<DeviceTensor> res1_bn2 = BatchNormOp<DeviceTensor> (128, 16384, 1e-05, nullptr, nullptr, &res1_bn2_weight, &res1_bn2_bias);
    LibDNNConvOp<DeviceTensor> res2_conv1 = LibDNNConvOp<DeviceTensor> ("res2_conv1_forward", 2097152, &res2_conv1_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {2048,32,1});
    BatchNormOp<DeviceTensor> res2_bn1 = BatchNormOp<DeviceTensor> (128, 16384, 1e-05, nullptr, nullptr, &res2_bn1_weight, &res2_bn1_bias);
    ReLUOp<DeviceTensor> res2_relu1 = ReLUOp<DeviceTensor> ( 0, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> res2_conv2 = LibDNNConvOp<DeviceTensor> ("res2_conv2_forward", 2097152, &res2_conv2_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {2048,32,1});
    BatchNormOp<DeviceTensor> res2_bn2 = BatchNormOp<DeviceTensor> (128, 16384, 1e-05, nullptr, nullptr, &res2_bn2_weight, &res2_bn2_bias);
    LibDNNConvOp<DeviceTensor> res3_conv1 = LibDNNConvOp<DeviceTensor> ("res3_conv1_forward", 2097152, &res3_conv1_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {2048,32,1});
    BatchNormOp<DeviceTensor> res3_bn1 = BatchNormOp<DeviceTensor> (128, 16384, 1e-05, nullptr, nullptr, &res3_bn1_weight, &res3_bn1_bias);
    ReLUOp<DeviceTensor> res3_relu1 = ReLUOp<DeviceTensor> ( 0, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> res3_conv2 = LibDNNConvOp<DeviceTensor> ("res3_conv2_forward", 2097152, &res3_conv2_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {2048,32,1});
    BatchNormOp<DeviceTensor> res3_bn2 = BatchNormOp<DeviceTensor> (128, 16384, 1e-05, nullptr, nullptr, &res3_bn2_weight, &res3_bn2_bias);
    LibDNNConvOp<DeviceTensor> res4_conv1 = LibDNNConvOp<DeviceTensor> ("res4_conv1_forward", 2097152, &res4_conv1_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {2048,32,1});
    BatchNormOp<DeviceTensor> res4_bn1 = BatchNormOp<DeviceTensor> (128, 16384, 1e-05, nullptr, nullptr, &res4_bn1_weight, &res4_bn1_bias);
    ReLUOp<DeviceTensor> res4_relu1 = ReLUOp<DeviceTensor> ( 0, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> res4_conv2 = LibDNNConvOp<DeviceTensor> ("res4_conv2_forward", 2097152, &res4_conv2_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {2048,32,1});
    BatchNormOp<DeviceTensor> res4_bn2 = BatchNormOp<DeviceTensor> (128, 16384, 1e-05, nullptr, nullptr, &res4_bn2_weight, &res4_bn2_bias);
    LibDNNConvOp<DeviceTensor> res5_conv1 = LibDNNConvOp<DeviceTensor> ("res5_conv1_forward", 2097152, &res5_conv1_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {2048,32,1});
    BatchNormOp<DeviceTensor> res5_bn1 = BatchNormOp<DeviceTensor> (128, 16384, 1e-05, nullptr, nullptr, &res5_bn1_weight, &res5_bn1_bias);
    ReLUOp<DeviceTensor> res5_relu1 = ReLUOp<DeviceTensor> ( 0, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> res5_conv2 = LibDNNConvOp<DeviceTensor> ("res5_conv2_forward", 2097152, &res5_conv2_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {2048,32,1});
    BatchNormOp<DeviceTensor> res5_bn2 = BatchNormOp<DeviceTensor> (128, 16384, 1e-05, nullptr, nullptr, &res5_bn2_weight, &res5_bn2_bias);
    LibDNNDeconvOp<DeviceTensor> deconv1 = LibDNNDeconvOp<DeviceTensor> ("deconv1_forward", 4194304, &deconv1_weight, &deconv1_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {8192,16,1});
    ELUOp<DeviceTensor> de_elu1 = ELUOp<DeviceTensor> ( 1, NOT_IN_PLACE );
    BatchNormOp<DeviceTensor> de_bn1 = BatchNormOp<DeviceTensor> (64, 65536, 1e-05, nullptr, nullptr, &de_bn1_weight, &de_bn1_bias);
    LibDNNDeconvOp<DeviceTensor> deconv2 = LibDNNDeconvOp<DeviceTensor> ("deconv2_forward", 8388608, &deconv2_weight, &deconv2_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {32768,8,1});
    ELUOp<DeviceTensor> de_elu2 = ELUOp<DeviceTensor> ( 1, NOT_IN_PLACE );
    BatchNormOp<DeviceTensor> de_bn2 = BatchNormOp<DeviceTensor> (32, 262144, 1e-05, nullptr, nullptr, &de_bn2_weight, &de_bn2_bias);
    LibDNNDeconvOp<DeviceTensor> deconv3 = LibDNNDeconvOp<DeviceTensor> ("deconv3_forward", 786432, &deconv3_weight, &deconv3_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {32768,4,1});
    TanHOp<DeviceTensor> de_tanh3 = TanHOp<DeviceTensor> ( NOT_IN_PLACE );

};


} //namespace hypertea