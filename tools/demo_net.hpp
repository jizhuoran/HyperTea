#include "hypertea/hypertea.hpp"

namespace hypertea {


// using DeviceTensor = TensorGPU<float>;


template <typename DeviceTensor>
class new_net {
public:

    new_net(const std::string &param_file) {

#ifdef USE_OPENCL

        OpenCLHandler::Get().build_opencl_math_code(false);
#endif //USE_OPENCL

        int weight_size = 7285260;
        unsigned char* all_weights = (unsigned char*) malloc(weight_size);

        FILE *f = fopen(param_file.c_str(), "rb");
        size_t read_size = fread(all_weights, 1, weight_size, f);
        if (read_size != weight_size) { 
            LOG(ERROR) << "Weight File Size Mismatch" << read_size << " and " << weight_size << std::endl;
        }
        fclose(f);

        param.copy_from_ptr((void*)all_weights);

        free(all_weights);

    }


    ~new_net() {}

    
    
    void inference( std::vector<float> &data_from_user, std::vector<float> &data_to_user) {
        

        auto data = DeviceTensor(data_from_user);

        auto temp = bn1(outplace_elu(conv1(data)));
        temp = bn2(outplace_elu(conv2(temp)));
        temp = bn3(outplace_elu(conv3(temp)));


        temp += res1_bn2(res1_conv2(outplace_relu(res1_bn1(res1_conv1(temp)))));
        temp += res2_bn2(res2_conv2(outplace_relu(res2_bn1(res2_conv1(temp)))));
        temp += res3_bn2(res3_conv2(outplace_relu(res3_bn1(res3_conv1(temp)))));
        temp += res4_bn2(res4_conv2(outplace_relu(res4_bn1(res4_conv1(temp)))));
        temp += res5_bn2(res5_conv2(outplace_relu(res5_bn1(res5_conv1(temp)))));
        

        temp = de_bn1(outplace_elu(deconv1(temp)));
        temp = de_bn2(outplace_elu(deconv2(temp)));
        temp = outplace_tanh(deconv3(temp));

        temp = (temp + 1) * 127.5;

        temp.copy_to_ptr((void*)data_to_user.data());

    }


private:

    DeviceTensor param = DeviceTensor(1821315);
     
    DeviceTensor conv1_bias = param.sub_view(0, 32);
    DeviceTensor conv1_weight = param.sub_view(32, 7776);
    DeviceTensor bn1_weight = param.sub_view(7808, 32);
    DeviceTensor bn1_bias = param.sub_view(7840, 32);
    DeviceTensor conv2_bias = param.sub_view(7872, 64);
    DeviceTensor conv2_weight = param.sub_view(7936, 32768);
    DeviceTensor bn2_weight = param.sub_view(40704, 64);
    DeviceTensor bn2_bias = param.sub_view(40768, 64);
    DeviceTensor conv3_bias = param.sub_view(40832, 128);
    DeviceTensor conv3_weight = param.sub_view(40960, 131072);
    DeviceTensor bn3_weight = param.sub_view(172032, 128);
    DeviceTensor bn3_bias = param.sub_view(172160, 128);
    DeviceTensor res1_conv1_weight = param.sub_view(172288, 147456);
    DeviceTensor res1_bn1_weight = param.sub_view(319744, 128);
    DeviceTensor res1_bn1_bias = param.sub_view(319872, 128);
    DeviceTensor res1_conv2_weight = param.sub_view(320000, 147456);
    DeviceTensor res1_bn2_weight = param.sub_view(467456, 128);
    DeviceTensor res1_bn2_bias = param.sub_view(467584, 128);
    DeviceTensor res2_conv1_weight = param.sub_view(467712, 147456);
    DeviceTensor res2_bn1_weight = param.sub_view(615168, 128);
    DeviceTensor res2_bn1_bias = param.sub_view(615296, 128);
    DeviceTensor res2_conv2_weight = param.sub_view(615424, 147456);
    DeviceTensor res2_bn2_weight = param.sub_view(762880, 128);
    DeviceTensor res2_bn2_bias = param.sub_view(763008, 128);
    DeviceTensor res3_conv1_weight = param.sub_view(763136, 147456);
    DeviceTensor res3_bn1_weight = param.sub_view(910592, 128);
    DeviceTensor res3_bn1_bias = param.sub_view(910720, 128);
    DeviceTensor res3_conv2_weight = param.sub_view(910848, 147456);
    DeviceTensor res3_bn2_weight = param.sub_view(1058304, 128);
    DeviceTensor res3_bn2_bias = param.sub_view(1058432, 128);
    DeviceTensor res4_conv1_weight = param.sub_view(1058560, 147456);
    DeviceTensor res4_bn1_weight = param.sub_view(1206016, 128);
    DeviceTensor res4_bn1_bias = param.sub_view(1206144, 128);
    DeviceTensor res4_conv2_weight = param.sub_view(1206272, 147456);
    DeviceTensor res4_bn2_weight = param.sub_view(1353728, 128);
    DeviceTensor res4_bn2_bias = param.sub_view(1353856, 128);
    DeviceTensor res5_conv1_weight = param.sub_view(1353984, 147456);
    DeviceTensor res5_bn1_weight = param.sub_view(1501440, 128);
    DeviceTensor res5_bn1_bias = param.sub_view(1501568, 128);
    DeviceTensor res5_conv2_weight = param.sub_view(1501696, 147456);
    DeviceTensor res5_bn2_weight = param.sub_view(1649152, 128);
    DeviceTensor res5_bn2_bias = param.sub_view(1649280, 128);
    DeviceTensor deconv1_bias = param.sub_view(1649408, 64);
    DeviceTensor deconv1_weight = param.sub_view(1649472, 131072);
    DeviceTensor de_bn1_weight = param.sub_view(1780544, 64);
    DeviceTensor de_bn1_bias = param.sub_view(1780608, 64);
    DeviceTensor deconv2_bias = param.sub_view(1780672, 32);
    DeviceTensor deconv2_weight = param.sub_view(1780704, 32768);
    DeviceTensor de_bn2_weight = param.sub_view(1813472, 32);
    DeviceTensor de_bn2_bias = param.sub_view(1813504, 32);
    DeviceTensor deconv3_bias = param.sub_view(1813536, 3);
    DeviceTensor deconv3_weight = param.sub_view(1813539, 7776);


    ConvolutionOp<DeviceTensor> conv1 = ConvolutionOp<DeviceTensor>(&conv1_weight, &conv1_bias, 1, false, std::vector<int> {9, 9}, std::vector<int> {1, 1}, std::vector<int> {4, 4}, std::vector<int> {1, 1}, std::vector<int> {2, 3, 512, 512}, std::vector<int> {2, 32, 512, 512});
    BatchNormOp<DeviceTensor> bn1 = BatchNormOp<DeviceTensor>(32, 262144, 1e-05, nullptr, nullptr, &bn1_weight, &bn1_bias);
    ConvolutionOp<DeviceTensor> conv2 = ConvolutionOp<DeviceTensor>(&conv2_weight, &conv2_bias, 1, false, std::vector<int> {4, 4}, std::vector<int> {2, 2}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 32, 512, 512}, std::vector<int> {2, 64, 256, 256});
    BatchNormOp<DeviceTensor> bn2 = BatchNormOp<DeviceTensor>(64, 65536, 1e-05, nullptr, nullptr, &bn2_weight, &bn2_bias);
    ConvolutionOp<DeviceTensor> conv3 = ConvolutionOp<DeviceTensor>(&conv3_weight, &conv3_bias, 1, false, std::vector<int> {4, 4}, std::vector<int> {2, 2}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 64, 256, 256}, std::vector<int> {2, 128, 128, 128});
    BatchNormOp<DeviceTensor> bn3 = BatchNormOp<DeviceTensor>(128, 16384, 1e-05, nullptr, nullptr, &bn3_weight, &bn3_bias);
    ConvolutionOp<DeviceTensor> res1_conv1 = ConvolutionOp<DeviceTensor>(&res1_conv1_weight, nullptr, 1, false, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 128, 128, 128}, std::vector<int> {2, 128, 128, 128});
    BatchNormOp<DeviceTensor> res1_bn1 = BatchNormOp<DeviceTensor>(128, 16384, 1e-05, nullptr, nullptr, &res1_bn1_weight, &res1_bn1_bias);
    ConvolutionOp<DeviceTensor> res1_conv2 = ConvolutionOp<DeviceTensor>(&res1_conv2_weight, nullptr, 1, false, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 128, 128, 128}, std::vector<int> {2, 128, 128, 128});
    BatchNormOp<DeviceTensor> res1_bn2 = BatchNormOp<DeviceTensor>(128, 16384, 1e-05, nullptr, nullptr, &res1_bn2_weight, &res1_bn2_bias);
    ConvolutionOp<DeviceTensor> res2_conv1 = ConvolutionOp<DeviceTensor>(&res2_conv1_weight, nullptr, 1, false, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 128, 128, 128}, std::vector<int> {2, 128, 128, 128});
    BatchNormOp<DeviceTensor> res2_bn1 = BatchNormOp<DeviceTensor>(128, 16384, 1e-05, nullptr, nullptr, &res2_bn1_weight, &res2_bn1_bias);
    ConvolutionOp<DeviceTensor> res2_conv2 = ConvolutionOp<DeviceTensor>(&res2_conv2_weight, nullptr, 1, false, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 128, 128, 128}, std::vector<int> {2, 128, 128, 128});
    BatchNormOp<DeviceTensor> res2_bn2 = BatchNormOp<DeviceTensor>(128, 16384, 1e-05, nullptr, nullptr, &res2_bn2_weight, &res2_bn2_bias);
    ConvolutionOp<DeviceTensor> res3_conv1 = ConvolutionOp<DeviceTensor>(&res3_conv1_weight, nullptr, 1, false, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 128, 128, 128}, std::vector<int> {2, 128, 128, 128});
    BatchNormOp<DeviceTensor> res3_bn1 = BatchNormOp<DeviceTensor>(128, 16384, 1e-05, nullptr, nullptr, &res3_bn1_weight, &res3_bn1_bias);
    ConvolutionOp<DeviceTensor> res3_conv2 = ConvolutionOp<DeviceTensor>(&res3_conv2_weight, nullptr, 1, false, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 128, 128, 128}, std::vector<int> {2, 128, 128, 128});
    BatchNormOp<DeviceTensor> res3_bn2 = BatchNormOp<DeviceTensor>(128, 16384, 1e-05, nullptr, nullptr, &res3_bn2_weight, &res3_bn2_bias);
    ConvolutionOp<DeviceTensor> res4_conv1 = ConvolutionOp<DeviceTensor>(&res4_conv1_weight, nullptr, 1, false, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 128, 128, 128}, std::vector<int> {2, 128, 128, 128});
    BatchNormOp<DeviceTensor> res4_bn1 = BatchNormOp<DeviceTensor>(128, 16384, 1e-05, nullptr, nullptr, &res4_bn1_weight, &res4_bn1_bias);
    ConvolutionOp<DeviceTensor> res4_conv2 = ConvolutionOp<DeviceTensor>(&res4_conv2_weight, nullptr, 1, false, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 128, 128, 128}, std::vector<int> {2, 128, 128, 128});
    BatchNormOp<DeviceTensor> res4_bn2 = BatchNormOp<DeviceTensor>(128, 16384, 1e-05, nullptr, nullptr, &res4_bn2_weight, &res4_bn2_bias);
    ConvolutionOp<DeviceTensor> res5_conv1 = ConvolutionOp<DeviceTensor>(&res5_conv1_weight, nullptr, 1, false, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 128, 128, 128}, std::vector<int> {2, 128, 128, 128});
    BatchNormOp<DeviceTensor> res5_bn1 = BatchNormOp<DeviceTensor>(128, 16384, 1e-05, nullptr, nullptr, &res5_bn1_weight, &res5_bn1_bias);
    ConvolutionOp<DeviceTensor> res5_conv2 = ConvolutionOp<DeviceTensor>(&res5_conv2_weight, nullptr, 1, false, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 128, 128, 128}, std::vector<int> {2, 128, 128, 128});
    BatchNormOp<DeviceTensor> res5_bn2 = BatchNormOp<DeviceTensor>(128, 16384, 1e-05, nullptr, nullptr, &res5_bn2_weight, &res5_bn2_bias);
    DeconvolutionOp<DeviceTensor> deconv1 = DeconvolutionOp<DeviceTensor>(&deconv1_weight, &deconv1_bias, 1, false, std::vector<int> {4, 4}, std::vector<int> {2, 2}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 128, 128, 128}, std::vector<int> {2, 64, 256, 256});
    BatchNormOp<DeviceTensor> de_bn1 = BatchNormOp<DeviceTensor>(64, 65536, 1e-05, nullptr, nullptr, &de_bn1_weight, &de_bn1_bias);
    DeconvolutionOp<DeviceTensor> deconv2 = DeconvolutionOp<DeviceTensor>(&deconv2_weight, &deconv2_bias, 1, false, std::vector<int> {4, 4}, std::vector<int> {2, 2}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 64, 256, 256}, std::vector<int> {2, 32, 512, 512});
    BatchNormOp<DeviceTensor> de_bn2 = BatchNormOp<DeviceTensor>(32, 262144, 1e-05, nullptr, nullptr, &de_bn2_weight, &de_bn2_bias); //de_bn2_bias
    DeconvolutionOp<DeviceTensor> deconv3 = DeconvolutionOp<DeviceTensor>(&deconv3_weight, &deconv3_bias, 1, false, std::vector<int> {9, 9}, std::vector<int> {1, 1}, std::vector<int> {4, 4}, std::vector<int> {1, 1}, std::vector<int> {2, 32, 512, 512}, std::vector<int> {2, 3, 512, 512});

};

 
} //namespace hypertea
        
