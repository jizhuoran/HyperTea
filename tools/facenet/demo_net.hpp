#include "hypertea/hypertea.hpp"
#include "kernels/conv_kernel.cl"

namespace hypertea {

template <typename DeviceTensor>
class facenet {

public:

    facenet(const std::string &param_file) { 

        compile_opencl_kernels(conv_opencl_funcs, " ");
        
        load_weight_to_tensor(param_file, param);

    }

    
    
    void inference( std::vector<float> &data_from_user, std::vector<int> &data_to_user) {
        
        auto x = DeviceTensor(data_from_user);


        x = relu1_1(conv1_1(x));

        x = x + relu1_3(conv1_3(relu1_2(conv1_2(x))));



        x = relu2_1(conv2_1(x));
        x = x + relu2_3(conv2_3(relu2_2(conv2_2(x))));
        x = x + relu2_5(conv2_5(relu2_4(conv2_4(x))));

        x = relu3_1(conv3_1(x));
        x = x + relu3_3(conv3_3(relu3_2(conv3_2(x))));
        x = x + relu3_5(conv3_5(relu3_4(conv3_4(x))));
        x = x + relu3_7(conv3_7(relu3_6(conv3_6(x))));
        x = x + relu3_9(conv3_9(relu3_8(conv3_8(x))));

        x = relu4_1(conv4_1(x));
        x = x + relu4_3(conv4_3(relu4_2(conv4_2(x))));

        x = fc5(x);
        x = fc6(x);

        data_to_user = x.argmax();

    }


private:
    
    DeviceTensor param = DeviceTensor(28095118);

     DeviceTensor conv1_1_bias = param.sub_view(0, 64);
     DeviceTensor conv1_1_weight = param.sub_view(64, 1728);
     DeviceTensor relu1_1_weight = param.sub_view(1792, 64);
     DeviceTensor conv1_2_bias = param.sub_view(1856, 64);
     DeviceTensor conv1_2_weight = param.sub_view(1920, 36864);
     DeviceTensor relu1_2_weight = param.sub_view(38784, 64);
     DeviceTensor conv1_3_bias = param.sub_view(38848, 64);
     DeviceTensor conv1_3_weight = param.sub_view(38912, 36864);
     DeviceTensor relu1_3_weight = param.sub_view(75776, 64);
     DeviceTensor conv2_1_bias = param.sub_view(75840, 128);
     DeviceTensor conv2_1_weight = param.sub_view(75968, 73728);
     DeviceTensor relu2_1_weight = param.sub_view(149696, 128);
     DeviceTensor conv2_2_bias = param.sub_view(149824, 128);
     DeviceTensor conv2_2_weight = param.sub_view(149952, 147456);
     DeviceTensor relu2_2_weight = param.sub_view(297408, 128);
     DeviceTensor conv2_3_bias = param.sub_view(297536, 128);
     DeviceTensor conv2_3_weight = param.sub_view(297664, 147456);
     DeviceTensor relu2_3_weight = param.sub_view(445120, 128);
     DeviceTensor conv2_4_bias = param.sub_view(445248, 128);
     DeviceTensor conv2_4_weight = param.sub_view(445376, 147456);
     DeviceTensor relu2_4_weight = param.sub_view(592832, 128);
     DeviceTensor conv2_5_bias = param.sub_view(592960, 128);
     DeviceTensor conv2_5_weight = param.sub_view(593088, 147456);
     DeviceTensor relu2_5_weight = param.sub_view(740544, 128);
     DeviceTensor conv3_1_bias = param.sub_view(740672, 256);
     DeviceTensor conv3_1_weight = param.sub_view(740928, 294912);
     DeviceTensor relu3_1_weight = param.sub_view(1035840, 256);
     DeviceTensor conv3_2_bias = param.sub_view(1036096, 256);
     DeviceTensor conv3_2_weight = param.sub_view(1036352, 589824);
     DeviceTensor relu3_2_weight = param.sub_view(1626176, 256);
     DeviceTensor conv3_3_bias = param.sub_view(1626432, 256);
     DeviceTensor conv3_3_weight = param.sub_view(1626688, 589824);
     DeviceTensor relu3_3_weight = param.sub_view(2216512, 256);
     DeviceTensor conv3_4_bias = param.sub_view(2216768, 256);
     DeviceTensor conv3_4_weight = param.sub_view(2217024, 589824);
     DeviceTensor relu3_4_weight = param.sub_view(2806848, 256);
     DeviceTensor conv3_5_bias = param.sub_view(2807104, 256);
     DeviceTensor conv3_5_weight = param.sub_view(2807360, 589824);
     DeviceTensor relu3_5_weight = param.sub_view(3397184, 256);
     DeviceTensor conv3_6_bias = param.sub_view(3397440, 256);
     DeviceTensor conv3_6_weight = param.sub_view(3397696, 589824);
     DeviceTensor relu3_6_weight = param.sub_view(3987520, 256);
     DeviceTensor conv3_7_bias = param.sub_view(3987776, 256);
     DeviceTensor conv3_7_weight = param.sub_view(3988032, 589824);
     DeviceTensor relu3_7_weight = param.sub_view(4577856, 256);
     DeviceTensor conv3_8_bias = param.sub_view(4578112, 256);
     DeviceTensor conv3_8_weight = param.sub_view(4578368, 589824);
     DeviceTensor relu3_8_weight = param.sub_view(5168192, 256);
     DeviceTensor conv3_9_bias = param.sub_view(5168448, 256);
     DeviceTensor conv3_9_weight = param.sub_view(5168704, 589824);
     DeviceTensor relu3_9_weight = param.sub_view(5758528, 256);
     DeviceTensor conv4_1_bias = param.sub_view(5758784, 512);
     DeviceTensor conv4_1_weight = param.sub_view(5759296, 1179648);
     DeviceTensor relu4_1_weight = param.sub_view(6938944, 512);
     DeviceTensor conv4_2_bias = param.sub_view(6939456, 512);
     DeviceTensor conv4_2_weight = param.sub_view(6939968, 2359296);
     DeviceTensor relu4_2_weight = param.sub_view(9299264, 512);
     DeviceTensor conv4_3_bias = param.sub_view(9299776, 512);
     DeviceTensor conv4_3_weight = param.sub_view(9300288, 2359296);
     DeviceTensor relu4_3_weight = param.sub_view(11659584, 512);
     DeviceTensor fc5_weight = param.sub_view(11660096, 11010048);
     DeviceTensor fc5_bias = param.sub_view(22670144, 512);
     DeviceTensor fc6_weight = param.sub_view(22670656, 5413888);
     DeviceTensor fc6_bias = param.sub_view(28084544, 10574);
    LibDNNConvOp<DeviceTensor> conv1_1 = LibDNNConvOp<DeviceTensor> ("conv1_1_forward", 172032, &conv1_1_weight, &conv1_1_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {336,16,1});
    PReLUOp<DeviceTensor> relu1_1 = PReLUOp<DeviceTensor> ( &relu1_1_weight, 64, 2688, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv1_2 = LibDNNConvOp<DeviceTensor> ("conv1_2_forward", 172032, &conv1_2_weight, &conv1_2_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {336,16,1});
    PReLUOp<DeviceTensor> relu1_2 = PReLUOp<DeviceTensor> ( &relu1_2_weight, 64, 2688, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv1_3 = LibDNNConvOp<DeviceTensor> ("conv1_3_forward", 172032, &conv1_3_weight, &conv1_3_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {336,16,1});
    PReLUOp<DeviceTensor> relu1_3 = PReLUOp<DeviceTensor> ( &relu1_3_weight, 64, 2688, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv2_1 = LibDNNConvOp<DeviceTensor> ("conv2_1_forward", 86016, &conv2_1_weight, &conv2_1_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,32,1});
    PReLUOp<DeviceTensor> relu2_1 = PReLUOp<DeviceTensor> ( &relu2_1_weight, 128, 672, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv2_2 = LibDNNConvOp<DeviceTensor> ("conv2_2_forward", 86016, &conv2_2_weight, &conv2_2_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,32,1});
    PReLUOp<DeviceTensor> relu2_2 = PReLUOp<DeviceTensor> ( &relu2_2_weight, 128, 672, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv2_3 = LibDNNConvOp<DeviceTensor> ("conv2_3_forward", 86016, &conv2_3_weight, &conv2_3_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,32,1});
    PReLUOp<DeviceTensor> relu2_3 = PReLUOp<DeviceTensor> ( &relu2_3_weight, 128, 672, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv2_4 = LibDNNConvOp<DeviceTensor> ("conv2_4_forward", 86016, &conv2_4_weight, &conv2_4_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,32,1});
    PReLUOp<DeviceTensor> relu2_4 = PReLUOp<DeviceTensor> ( &relu2_4_weight, 128, 672, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv2_5 = LibDNNConvOp<DeviceTensor> ("conv2_5_forward", 86016, &conv2_5_weight, &conv2_5_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,32,1});
    PReLUOp<DeviceTensor> relu2_5 = PReLUOp<DeviceTensor> ( &relu2_5_weight, 128, 672, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv3_1 = LibDNNConvOp<DeviceTensor> ("conv3_1_forward", 43008, &conv3_1_weight, &conv3_1_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,64,1});
    PReLUOp<DeviceTensor> relu3_1 = PReLUOp<DeviceTensor> ( &relu3_1_weight, 256, 168, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv3_2 = LibDNNConvOp<DeviceTensor> ("conv3_2_forward", 43008, &conv3_2_weight, &conv3_2_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,64,1});
    PReLUOp<DeviceTensor> relu3_2 = PReLUOp<DeviceTensor> ( &relu3_2_weight, 256, 168, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv3_3 = LibDNNConvOp<DeviceTensor> ("conv3_3_forward", 43008, &conv3_3_weight, &conv3_3_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,64,1});
    PReLUOp<DeviceTensor> relu3_3 = PReLUOp<DeviceTensor> ( &relu3_3_weight, 256, 168, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv3_4 = LibDNNConvOp<DeviceTensor> ("conv3_4_forward", 43008, &conv3_4_weight, &conv3_4_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,64,1});
    PReLUOp<DeviceTensor> relu3_4 = PReLUOp<DeviceTensor> ( &relu3_4_weight, 256, 168, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv3_5 = LibDNNConvOp<DeviceTensor> ("conv3_5_forward", 43008, &conv3_5_weight, &conv3_5_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,64,1});
    PReLUOp<DeviceTensor> relu3_5 = PReLUOp<DeviceTensor> ( &relu3_5_weight, 256, 168, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv3_6 = LibDNNConvOp<DeviceTensor> ("conv3_6_forward", 43008, &conv3_6_weight, &conv3_6_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,64,1});
    PReLUOp<DeviceTensor> relu3_6 = PReLUOp<DeviceTensor> ( &relu3_6_weight, 256, 168, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv3_7 = LibDNNConvOp<DeviceTensor> ("conv3_7_forward", 43008, &conv3_7_weight, &conv3_7_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,64,1});
    PReLUOp<DeviceTensor> relu3_7 = PReLUOp<DeviceTensor> ( &relu3_7_weight, 256, 168, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv3_8 = LibDNNConvOp<DeviceTensor> ("conv3_8_forward", 43008, &conv3_8_weight, &conv3_8_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,64,1});
    PReLUOp<DeviceTensor> relu3_8 = PReLUOp<DeviceTensor> ( &relu3_8_weight, 256, 168, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv3_9 = LibDNNConvOp<DeviceTensor> ("conv3_9_forward", 43008, &conv3_9_weight, &conv3_9_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,64,1});
    PReLUOp<DeviceTensor> relu3_9 = PReLUOp<DeviceTensor> ( &relu3_9_weight, 256, 168, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv4_1 = LibDNNConvOp<DeviceTensor> ("conv4_1_forward", 21504, &conv4_1_weight, &conv4_1_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {16,128,1});
    PReLUOp<DeviceTensor> relu4_1 = PReLUOp<DeviceTensor> ( &relu4_1_weight, 512, 42, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv4_2 = LibDNNConvOp<DeviceTensor> ("conv4_2_forward", 21504, &conv4_2_weight, &conv4_2_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {16,128,1});
    PReLUOp<DeviceTensor> relu4_2 = PReLUOp<DeviceTensor> ( &relu4_2_weight, 512, 42, NOT_IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv4_3 = LibDNNConvOp<DeviceTensor> ("conv4_3_forward", 21504, &conv4_3_weight, &conv4_3_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {16,128,1});
    PReLUOp<DeviceTensor> relu4_3 = PReLUOp<DeviceTensor> ( &relu4_3_weight, 512, 42, NOT_IN_PLACE );
    LinearOp<DeviceTensor> fc5 = LinearOp<DeviceTensor> ( &fc5_weight, &fc5_bias, 21504, 512 );
    LinearOp<DeviceTensor> fc6 = LinearOp<DeviceTensor> ( &fc6_weight, &fc6_bias, 512, 10574 );

};


} //namespace hypertea