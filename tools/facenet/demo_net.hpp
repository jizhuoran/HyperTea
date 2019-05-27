#include "hypertea/hypertea.hpp"
#include "kernels/conv_kernel.cl"

namespace hypertea {

template <typename DeviceTensor>
class facenet {

public:

    facenet(const std::string &param_file) { 

        // compile_opencl_kernels(conv_opencl_funcs, " ");
        
        load_opencl_kernels("/sdcard/hypertea_ws/facenet/prebuilt_math_program", "/sdcard/hypertea_ws/facenet/prebuilt_conv_program", " ");

        auto all_weights = load_weights<float>(param_file, 28095118);


        conv1_1_bias.copy_from_ptr(all_weights + 0);
        conv1_1_weight.copy_from_ptr(all_weights + 64);
        relu1_1_weight.copy_from_ptr(all_weights + 1792);
        conv1_2_bias.copy_from_ptr(all_weights + 1856);
        conv1_2_weight.copy_from_ptr(all_weights + 1920);
        relu1_2_weight.copy_from_ptr(all_weights + 38784);
        conv1_3_bias.copy_from_ptr(all_weights + 38848);
        conv1_3_weight.copy_from_ptr(all_weights + 38912);
        relu1_3_weight.copy_from_ptr(all_weights + 75776);
        conv2_1_bias.copy_from_ptr(all_weights + 75840);
        conv2_1_weight.copy_from_ptr(all_weights + 75968);
        relu2_1_weight.copy_from_ptr(all_weights + 149696);
        conv2_2_bias.copy_from_ptr(all_weights + 149824);
        conv2_2_weight.copy_from_ptr(all_weights + 149952);
        relu2_2_weight.copy_from_ptr(all_weights + 297408);
        conv2_3_bias.copy_from_ptr(all_weights + 297536);
        conv2_3_weight.copy_from_ptr(all_weights + 297664);
        relu2_3_weight.copy_from_ptr(all_weights + 445120);
        conv2_4_bias.copy_from_ptr(all_weights + 445248);
        conv2_4_weight.copy_from_ptr(all_weights + 445376);
        relu2_4_weight.copy_from_ptr(all_weights + 592832);
        conv2_5_bias.copy_from_ptr(all_weights + 592960);
        conv2_5_weight.copy_from_ptr(all_weights + 593088);
        relu2_5_weight.copy_from_ptr(all_weights + 740544);
        conv3_1_bias.copy_from_ptr(all_weights + 740672);
        conv3_1_weight.copy_from_ptr(all_weights + 740928);
        relu3_1_weight.copy_from_ptr(all_weights + 1035840);
        conv3_2_bias.copy_from_ptr(all_weights + 1036096);
        conv3_2_weight.copy_from_ptr(all_weights + 1036352);
        relu3_2_weight.copy_from_ptr(all_weights + 1626176);
        conv3_3_bias.copy_from_ptr(all_weights + 1626432);
        conv3_3_weight.copy_from_ptr(all_weights + 1626688);
        relu3_3_weight.copy_from_ptr(all_weights + 2216512);
        conv3_4_bias.copy_from_ptr(all_weights + 2216768);
        conv3_4_weight.copy_from_ptr(all_weights + 2217024);
        relu3_4_weight.copy_from_ptr(all_weights + 2806848);
        conv3_5_bias.copy_from_ptr(all_weights + 2807104);
        conv3_5_weight.copy_from_ptr(all_weights + 2807360);
        relu3_5_weight.copy_from_ptr(all_weights + 3397184);
        conv3_6_bias.copy_from_ptr(all_weights + 3397440);
        conv3_6_weight.copy_from_ptr(all_weights + 3397696);
        relu3_6_weight.copy_from_ptr(all_weights + 3987520);
        conv3_7_bias.copy_from_ptr(all_weights + 3987776);
        conv3_7_weight.copy_from_ptr(all_weights + 3988032);
        relu3_7_weight.copy_from_ptr(all_weights + 4577856);
        conv3_8_bias.copy_from_ptr(all_weights + 4578112);
        conv3_8_weight.copy_from_ptr(all_weights + 4578368);
        relu3_8_weight.copy_from_ptr(all_weights + 5168192);
        conv3_9_bias.copy_from_ptr(all_weights + 5168448);
        conv3_9_weight.copy_from_ptr(all_weights + 5168704);
        relu3_9_weight.copy_from_ptr(all_weights + 5758528);
        conv4_1_bias.copy_from_ptr(all_weights + 5758784);
        conv4_1_weight.copy_from_ptr(all_weights + 5759296);
        relu4_1_weight.copy_from_ptr(all_weights + 6938944);
        conv4_2_bias.copy_from_ptr(all_weights + 6939456);
        conv4_2_weight.copy_from_ptr(all_weights + 6939968);
        relu4_2_weight.copy_from_ptr(all_weights + 9299264);
        conv4_3_bias.copy_from_ptr(all_weights + 9299776);
        conv4_3_weight.copy_from_ptr(all_weights + 9300288);
        relu4_3_weight.copy_from_ptr(all_weights + 11659584);
        fc5_weight.copy_from_ptr(all_weights + 11660096);
        fc5_bias.copy_from_ptr(all_weights + 22670144);
        fc6_weight.copy_from_ptr(all_weights + 22670656);
        fc6_bias.copy_from_ptr(all_weights + 28084544);


        free(all_weights);

    }

    
    
    void inference( std::vector<float> &data_from_user, std::vector<int> &data_to_user) {
        
        std::cout << "DEBUG: come to here!!!" << std::endl;

        auto x = DeviceTensor(data_from_user);

        std::cout << "DEBUG: come to here 1!!!" << std::endl;

        x = conv1_1(x);

        std::cout << "DEBUG: come to here 1.5!!!" << std::endl;

        x = relu1_1(x);

        std::cout << "DEBUG: come to here 1.6!!!" << std::endl;

        x = conv1_2(x);

        std::cout << "DEBUG: come to here 1.7!!!" << std::endl;


        x = x + relu1_3(conv1_3(relu1_2(x)));

        std::cout << "DEBUG: come to here 2!!!" << std::endl;


        x = relu2_1(conv2_1(x));
        x = x + relu2_3(conv2_3(relu2_2(conv2_2(x))));
        x = x + relu2_5(conv2_5(relu2_4(conv2_4(x))));

        std::cout << "DEBUG: come to here 3!!!" << std::endl;


        x = relu3_1(conv3_1(x));
        x = x + relu3_3(conv3_3(relu3_2(conv3_2(x))));
        x = x + relu3_5(conv3_5(relu3_4(conv3_4(x))));
        x = x + relu3_7(conv3_7(relu3_6(conv3_6(x))));
        x = x + relu3_9(conv3_9(relu3_8(conv3_8(x))));

        std::cout << "DEBUG: come to here 4!!!" << std::endl;


        x = relu4_1(conv4_1(x));
        x = x + relu4_3(conv4_3(relu4_2(conv4_2(x))));

        std::cout << "DEBUG: come to here 5!!!" << std::endl;


        x = fc5(x);
        x = fc6(x);



        data_to_user = x.argmax();

        std::cout << "DEBUG: come to here 6!!!" << std::endl;

    }


private:
    

    DeviceTensor conv1_1_bias = DeviceTensor(64);
    DeviceTensor conv1_1_weight = DeviceTensor(1728);
    DeviceTensor relu1_1_weight = DeviceTensor(64);
    DeviceTensor conv1_2_bias = DeviceTensor(64);
    DeviceTensor conv1_2_weight = DeviceTensor(36864);
    DeviceTensor relu1_2_weight = DeviceTensor(64);
    DeviceTensor conv1_3_bias = DeviceTensor(64);
    DeviceTensor conv1_3_weight = DeviceTensor(36864);
    DeviceTensor relu1_3_weight = DeviceTensor(64);
    DeviceTensor conv2_1_bias = DeviceTensor(128);
    DeviceTensor conv2_1_weight = DeviceTensor(73728);
    DeviceTensor relu2_1_weight = DeviceTensor(128);
    DeviceTensor conv2_2_bias = DeviceTensor(128);
    DeviceTensor conv2_2_weight = DeviceTensor(147456);
    DeviceTensor relu2_2_weight = DeviceTensor(128);
    DeviceTensor conv2_3_bias = DeviceTensor(128);
    DeviceTensor conv2_3_weight = DeviceTensor(147456);
    DeviceTensor relu2_3_weight = DeviceTensor(128);
    DeviceTensor conv2_4_bias = DeviceTensor(128);
    DeviceTensor conv2_4_weight = DeviceTensor(147456);
    DeviceTensor relu2_4_weight = DeviceTensor(128);
    DeviceTensor conv2_5_bias = DeviceTensor(128);
    DeviceTensor conv2_5_weight = DeviceTensor(147456);
    DeviceTensor relu2_5_weight = DeviceTensor(128);
    DeviceTensor conv3_1_bias = DeviceTensor(256);
    DeviceTensor conv3_1_weight = DeviceTensor(294912);
    DeviceTensor relu3_1_weight = DeviceTensor(256);
    DeviceTensor conv3_2_bias = DeviceTensor(256);
    DeviceTensor conv3_2_weight = DeviceTensor(589824);
    DeviceTensor relu3_2_weight = DeviceTensor(256);
    DeviceTensor conv3_3_bias = DeviceTensor(256);
    DeviceTensor conv3_3_weight = DeviceTensor(589824);
    DeviceTensor relu3_3_weight = DeviceTensor(256);
    DeviceTensor conv3_4_bias = DeviceTensor(256);
    DeviceTensor conv3_4_weight = DeviceTensor(589824);
    DeviceTensor relu3_4_weight = DeviceTensor(256);
    DeviceTensor conv3_5_bias = DeviceTensor(256);
    DeviceTensor conv3_5_weight = DeviceTensor(589824);
    DeviceTensor relu3_5_weight = DeviceTensor(256);
    DeviceTensor conv3_6_bias = DeviceTensor(256);
    DeviceTensor conv3_6_weight = DeviceTensor(589824);
    DeviceTensor relu3_6_weight = DeviceTensor(256);
    DeviceTensor conv3_7_bias = DeviceTensor(256);
    DeviceTensor conv3_7_weight = DeviceTensor(589824);
    DeviceTensor relu3_7_weight = DeviceTensor(256);
    DeviceTensor conv3_8_bias = DeviceTensor(256);
    DeviceTensor conv3_8_weight = DeviceTensor(589824);
    DeviceTensor relu3_8_weight = DeviceTensor(256);
    DeviceTensor conv3_9_bias = DeviceTensor(256);
    DeviceTensor conv3_9_weight = DeviceTensor(589824);
    DeviceTensor relu3_9_weight = DeviceTensor(256);
    DeviceTensor conv4_1_bias = DeviceTensor(512);
    DeviceTensor conv4_1_weight = DeviceTensor(1179648);
    DeviceTensor relu4_1_weight = DeviceTensor(512);
    DeviceTensor conv4_2_bias = DeviceTensor(512);
    DeviceTensor conv4_2_weight = DeviceTensor(2359296);
    DeviceTensor relu4_2_weight = DeviceTensor(512);
    DeviceTensor conv4_3_bias = DeviceTensor(512);
    DeviceTensor conv4_3_weight = DeviceTensor(2359296);
    DeviceTensor relu4_3_weight = DeviceTensor(512);
    DeviceTensor fc5_weight = DeviceTensor(11010048);
    DeviceTensor fc5_bias = DeviceTensor(512);
    DeviceTensor fc6_weight = DeviceTensor(5413888);
    DeviceTensor fc6_bias = DeviceTensor(10574);

     
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