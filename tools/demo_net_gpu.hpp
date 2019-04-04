#include "hypertea/hypertea.hpp"
#include "bn_opencl.hpp"
#include "conv_opencl.hpp"

namespace hypertea {

class new_net {
public:

     

    new_net(const std::string &param_file) {

        int weight_size = 7285260;
        unsigned char* all_weights = (unsigned char*) malloc(weight_size);

        FILE *f = fopen(param_file.c_str(), "rb");
        size_t read_size = fread(all_weights, 1, weight_size, f);
        if (read_size != weight_size) { 
            LOG(ERROR) << "Weight File Size Mismatch" << read_size << " and " << weight_size << std::endl;
        }
        fclose(f);

        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, param.mutable_data(), CL_TRUE, 0, 7285260, all_weights, 0, NULL, NULL));

        free(all_weights);

        OpenCLHandler::Get().build_opencl_math_code(false);
        OpenCLHandler::Get().build_opencl_program(conv_opencl_funcs, OpenCLHandler::Get().conv_program);
        OpenCLHandler::Get().build_opencl_program(bn_opencl_funcs, OpenCLHandler::Get().bn_program);

    }


    ~new_net() {}

    
    
    void inference( std::vector<float> &data_from_user, std::vector<float> &data_to_user) {
        
        TensorGPU<float> data(data_from_user);

        auto temp = bn1(gpu_elu(conv1(data)));
        temp = bn2(gpu_elu(conv2(temp)));
        temp = bn3(gpu_elu(conv3(temp)));


        temp += res1_bn2(res1_conv2(gpu_relu(res1_bn1(res1_conv1(temp)))));
        temp += res2_bn2(res2_conv2(gpu_relu(res2_bn1(res2_conv1(temp)))));
        temp += res3_bn2(res3_conv2(gpu_relu(res3_bn1(res3_conv1(temp)))));
        temp += res4_bn2(res4_conv2(gpu_relu(res4_bn1(res4_conv1(temp)))));
        temp += res5_bn2(res5_conv2(gpu_relu(res5_bn1(res5_conv1(temp)))));
        

        temp = de_bn1(gpu_elu(deconv1(temp)));
        temp = de_bn2(gpu_elu(deconv2(temp)));
        temp = gpu_tanh(deconv3(temp));

        temp = (temp + 1) * 127.5;

        OPENCL_CHECK(clEnqueueReadBuffer(OpenCLHandler::Get().commandQueue, temp.immutable_data(), CL_TRUE, 0, data_to_user.size() * sizeof(data_to_user[0]), data_to_user.data(), 0, NULL, NULL));

    }


private:

    TensorGPU<float> param = TensorGPU<float>(7285260);
     

    // TensorGPU<float> conv1_bias = param.sub_view(0, 32);
    // TensorGPU<float> conv1_weight = param.sub_view(32, 7776);
    TensorGPU<float> bn1_weight = param.sub_view(7808, 32);
    TensorGPU<float> bn1_bias = param.sub_view(7840, 32);
    TensorGPU<float> conv2_bias = param.sub_view(7872, 64);
    TensorGPU<float> conv2_weight = param.sub_view(7936, 32768);
    TensorGPU<float> bn2_weight = param.sub_view(40704, 64);
    TensorGPU<float> bn2_bias = param.sub_view(40768, 64);
    TensorGPU<float> conv3_bias = param.sub_view(40832, 128);
    TensorGPU<float> conv3_weight = param.sub_view(40960, 131072);
    TensorGPU<float> bn3_weight = param.sub_view(172032, 128);
    TensorGPU<float> bn3_bias = param.sub_view(172160, 128);
    TensorGPU<float> res1_conv1_weight = param.sub_view(172288, 147456);
    TensorGPU<float> res1_bn1_weight = param.sub_view(319744, 128);
    TensorGPU<float> res1_bn1_bias = param.sub_view(319872, 128);
    TensorGPU<float> res1_conv2_weight = param.sub_view(320000, 147456);
    TensorGPU<float> res1_bn2_weight = param.sub_view(467456, 128);
    TensorGPU<float> res1_bn2_bias = param.sub_view(467584, 128);
    TensorGPU<float> res2_conv1_weight = param.sub_view(467712, 147456);
    TensorGPU<float> res2_bn1_weight = param.sub_view(615168, 128);
    TensorGPU<float> res2_bn1_bias = param.sub_view(615296, 128);
    TensorGPU<float> res2_conv2_weight = param.sub_view(615424, 147456);
    TensorGPU<float> res2_bn2_weight = param.sub_view(762880, 128);
    TensorGPU<float> res2_bn2_bias = param.sub_view(763008, 128);
    TensorGPU<float> res3_conv1_weight = param.sub_view(763136, 147456);
    TensorGPU<float> res3_bn1_weight = param.sub_view(910592, 128);
    TensorGPU<float> res3_bn1_bias = param.sub_view(910720, 128);
    TensorGPU<float> res3_conv2_weight = param.sub_view(910848, 147456);
    TensorGPU<float> res3_bn2_weight = param.sub_view(1058304, 128);
    TensorGPU<float> res3_bn2_bias = param.sub_view(1058432, 128);
    TensorGPU<float> res4_conv1_weight = param.sub_view(1058560, 147456);
    TensorGPU<float> res4_bn1_weight = param.sub_view(1206016, 128);
    TensorGPU<float> res4_bn1_bias = param.sub_view(1206144, 128);
    TensorGPU<float> res4_conv2_weight = param.sub_view(1206272, 147456);
    TensorGPU<float> res4_bn2_weight = param.sub_view(1353728, 128);
    TensorGPU<float> res4_bn2_bias = param.sub_view(1353856, 128);
    TensorGPU<float> res5_conv1_weight = param.sub_view(1353984, 147456);
    TensorGPU<float> res5_bn1_weight = param.sub_view(1501440, 128);
    TensorGPU<float> res5_bn1_bias = param.sub_view(1501568, 128);
    TensorGPU<float> res5_conv2_weight = param.sub_view(1501696, 147456);
    TensorGPU<float> res5_bn2_weight = param.sub_view(1649152, 128);
    TensorGPU<float> res5_bn2_bias = param.sub_view(1649280, 128);
    TensorGPU<float> deconv1_bias = param.sub_view(1649408, 64);
    TensorGPU<float> deconv1_weight = param.sub_view(1649472, 131072);
    TensorGPU<float> de_bn1_weight = param.sub_view(1780544, 64);
    TensorGPU<float> de_bn1_bias = param.sub_view(1780608, 64);
    TensorGPU<float> deconv2_bias = param.sub_view(1780672, 32);
    TensorGPU<float> deconv2_weight = param.sub_view(1780704, 32768);
    TensorGPU<float> de_bn2_weight = param.sub_view(1813472, 32);
    TensorGPU<float> de_bn2_bias = param.sub_view(1813504, 32);
    TensorGPU<float> deconv3_bias = param.sub_view(1813536, 3);
    TensorGPU<float> deconv3_weight = param.sub_view(1813539, 7776);


    ConvolutionOp_GPU<float> conv1 = ConvolutionOp_GPU<float> (
        "conv1_forward", 8388608, 
        param.sub_view(32, 7776), param.sub_view(0, 32), 
        std::vector<int> {16,4,1}, std::vector<int> {32768,8,1}
    );

    BatchNormOp_GPU<float> bn1 = BatchNormOp_GPU<float> (8388608, 1, 32, 1e-05, 1, false, TensorGPU<float>(32), TensorGPU<float>(32), bn1_weight, bn1_bias);
    ConvolutionOp_GPU<float> conv2 = ConvolutionOp_GPU<float> ("conv2_forward", 4194304, conv2_weight, conv2_bias, std::vector<int> {16,4,1}, std::vector<int> {8192,16,1});
    BatchNormOp_GPU<float> bn2 = BatchNormOp_GPU<float> (4194304, 1, 64, 1e-05, 1, false, TensorGPU<float>(64), TensorGPU<float>(64), bn2_weight, bn2_bias);
    ConvolutionOp_GPU<float> conv3 = ConvolutionOp_GPU<float> ("conv3_forward", 2097152, conv3_weight, conv3_bias, std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<float> bn3 = BatchNormOp_GPU<float> (2097152, 1, 128, 1e-05, 1, false, TensorGPU<float>(128), TensorGPU<float>(128), bn3_weight, bn3_bias);
    ConvolutionOp_GPU<float> res1_conv1 = ConvolutionOp_GPU<float> ("res1_conv1_forward", 2097152, res1_conv1_weight, TensorGPU<float>(0), std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<float> res1_bn1 = BatchNormOp_GPU<float> (2097152, 1, 128, 1e-05, 1, false, TensorGPU<float>(128), TensorGPU<float>(128), res1_bn1_weight, res1_bn1_bias);
    ConvolutionOp_GPU<float> res1_conv2 = ConvolutionOp_GPU<float> ("res1_conv2_forward", 2097152, res1_conv2_weight, TensorGPU<float>(0), std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<float> res1_bn2 = BatchNormOp_GPU<float> (2097152, 1, 128, 1e-05, 1, false, TensorGPU<float>(128), TensorGPU<float>(128), res1_bn2_weight, res1_bn2_bias);
    ConvolutionOp_GPU<float> res2_conv1 = ConvolutionOp_GPU<float> ("res2_conv1_forward", 2097152, res2_conv1_weight, TensorGPU<float>(0), std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<float> res2_bn1 = BatchNormOp_GPU<float> (2097152, 1, 128, 1e-05, 1, false, TensorGPU<float>(128), TensorGPU<float>(128), res2_bn1_weight, res2_bn1_bias);
    ConvolutionOp_GPU<float> res2_conv2 = ConvolutionOp_GPU<float> ("res2_conv2_forward", 2097152, res2_conv2_weight, TensorGPU<float>(0), std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<float> res2_bn2 = BatchNormOp_GPU<float> (2097152, 1, 128, 1e-05, 1, false, TensorGPU<float>(128), TensorGPU<float>(128), res2_bn2_weight, res2_bn2_bias);
    ConvolutionOp_GPU<float> res3_conv1 = ConvolutionOp_GPU<float> ("res3_conv1_forward", 2097152, res3_conv1_weight, TensorGPU<float>(0), std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<float> res3_bn1 = BatchNormOp_GPU<float> (2097152, 1, 128, 1e-05, 1, false, TensorGPU<float>(128), TensorGPU<float>(128), res3_bn1_weight, res3_bn1_bias);
    ConvolutionOp_GPU<float> res3_conv2 = ConvolutionOp_GPU<float> ("res3_conv2_forward", 2097152, res3_conv2_weight, TensorGPU<float>(0), std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<float> res3_bn2 = BatchNormOp_GPU<float> (2097152, 1, 128, 1e-05, 1, false, TensorGPU<float>(128), TensorGPU<float>(128), res3_bn2_weight, res3_bn2_bias);
    ConvolutionOp_GPU<float> res4_conv1 = ConvolutionOp_GPU<float> ("res4_conv1_forward", 2097152, res4_conv1_weight, TensorGPU<float>(0), std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<float> res4_bn1 = BatchNormOp_GPU<float> (2097152, 1, 128, 1e-05, 1, false, TensorGPU<float>(128), TensorGPU<float>(128), res4_bn1_weight, res4_bn1_bias);
    ConvolutionOp_GPU<float> res4_conv2 = ConvolutionOp_GPU<float> ("res4_conv2_forward", 2097152, res4_conv2_weight, TensorGPU<float>(0), std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<float> res4_bn2 = BatchNormOp_GPU<float> (2097152, 1, 128, 1e-05, 1, false, TensorGPU<float>(128), TensorGPU<float>(128), res4_bn2_weight, res4_bn2_bias);
    ConvolutionOp_GPU<float> res5_conv1 = ConvolutionOp_GPU<float> ("res5_conv1_forward", 2097152, res5_conv1_weight, TensorGPU<float>(0), std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<float> res5_bn1 = BatchNormOp_GPU<float> (2097152, 1, 128, 1e-05, 1, false, TensorGPU<float>(128), TensorGPU<float>(128), res5_bn1_weight, res5_bn1_bias);
    ConvolutionOp_GPU<float> res5_conv2 = ConvolutionOp_GPU<float> ("res5_conv2_forward", 2097152, res5_conv2_weight, TensorGPU<float>(0), std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<float> res5_bn2 = BatchNormOp_GPU<float> (2097152, 1, 128, 1e-05, 1, false, TensorGPU<float>(128), TensorGPU<float>(128), res5_bn2_weight, res5_bn2_bias);
    DeconvolutionOp_GPU<float> deconv1 = DeconvolutionOp_GPU<float> ("deconv1_forward", 4194304, deconv1_weight, deconv1_bias, std::vector<int> {16,4,1}, std::vector<int> {8192,16,1});
    BatchNormOp_GPU<float> de_bn1 = BatchNormOp_GPU<float> (4194304, 1, 64, 1e-05, 1, false, TensorGPU<float>(64), TensorGPU<float>(64), de_bn1_weight, de_bn1_bias);
    DeconvolutionOp_GPU<float> deconv2 = DeconvolutionOp_GPU<float> ("deconv2_forward", 8388608, deconv2_weight, deconv2_bias, std::vector<int> {16,4,1}, std::vector<int> {32768,8,1});
    BatchNormOp_GPU<float> de_bn2 = BatchNormOp_GPU<float> (8388608, 1, 32, 1e-05, 1, false, TensorGPU<float>(32), TensorGPU<float>(32), de_bn2_weight, de_bn2_bias); //de_bn2_bias
    DeconvolutionOp_GPU<float> deconv3 = DeconvolutionOp_GPU<float> ("deconv3_forward", 786432, deconv3_weight, deconv3_bias, std::vector<int> {16,4,1}, std::vector<int> {32768,4,1});

};

 





















} //namespace hypertea
        
