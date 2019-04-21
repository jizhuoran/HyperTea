#include "hypertea/hypertea.hpp"
#include "kernels/conv_kernel.cl"

namespace hypertea {

using DeviceTensor = TensorGPU<float>;

class DetectedInfo
{
public:
    DetectedInfo(
        float x1, float y1, 
        float x2, float y2,
        float object_conf,
        float pos_conf,
        int object_index
    ) : x1_(x1), y1_(y1), 
        x2_(x2), y2_(y2), 
        object_conf_(object_conf), 
        pos_conf_(pos_conf), 
        object_index_(object_index) {}
   

    float x1_;
    float y1_; 
    float x2_;
    float y2_;
    float object_conf_;
    float pos_conf_;
    int object_index_;
};


void predict_transform(
    DeviceTensor prediction, 
    int batch_size, 
    int stride, 
    int grid_size, 
    std::vector<float> anchors, 
    int num_classes,
    float confidence,
    std::vector<DetectedInfo>& results) {


    int bbox_attrs = num_classes + 5;

    int num_anchors = anchors.size() / 2;
    
    int grid_square = grid_size * grid_size;

    for (int i = 0; i < anchors.size(); ++i) {
        anchors[i] /= stride;
    }

    float confidence_inv_sigmoid = log(confidence / (1 - confidence));


    auto cpu_data = prediction.debug_gtest_cpu_data();

    std::vector<int> pos_index;
    std::vector<int> anchor_index;


    for (int n = 0; n < num_anchors; ++n) {
        int anchor_offset = (n * bbox_attrs + 4) * grid_square;
        for (int i = 0; i < grid_square; ++i) {
            if (cpu_data.get()[anchor_offset + i] > confidence_inv_sigmoid) {
                pos_index.push_back(i);
                anchor_index.push_back(n);
            }
        }
    }

    int out_num = pos_index.size();


    if (out_num == 0) { return; }


    TensorCPU<float> output(out_num * bbox_attrs);
    auto output_data = output.mutable_data();


    for (int i = 0; i < bbox_attrs; ++i) {
        for (int n = 0; n < out_num; ++n) {
            output_data[i * out_num + n] = cpu_data.get()[(anchor_index[n] * bbox_attrs + i) * grid_square + pos_index[n]];
        }
        
    }


    auto xy1 = output.sub_view(0, out_num*2); inplace_sigmoid(xy1);
    auto xy2 = output.sub_view(out_num*2, out_num*2); inplace_exp(xy2);
    auto prob = output.sub_view(out_num*4, out_num); inplace_sigmoid(prob);
    

    auto uname_remain = output.sub_view(out_num*5, out_num*num_classes); //inplace_sigmoid(uname_remain);


    auto xy1_data = xy1.mutable_data();
    auto xy2_data = xy2.mutable_data();


    for (int i = 0; i < out_num; ++i) {

        xy1_data[i] = (xy1_data[i] + (pos_index[i] % grid_size))*stride;
        xy1_data[i + out_num] = (xy1_data[i + out_num] + (pos_index[i] / grid_size))*stride;

        xy2_data[i] *= (stride * anchors[anchor_index[i] * 2]);
        xy2_data[i + out_num] *= (stride * anchors[anchor_index[i] * 2 + 1]);

    }



    TensorCPU<float> box_a = xy1 - xy2 / 2;
    xy1.copy_data(box_a);

    box_a += xy2;
    xy2.copy_data(box_a);


    uname_remain = uname_remain.transpose_hw(out_num, num_classes);
    auto max_conf = batched_argmax(uname_remain, num_classes);

    for (int i = 0; i < out_num; ++i) {
        results.push_back(DetectedInfo(
            output_data[0 * out_num + i], 
            output_data[1 * out_num + i], 
            output_data[2 * out_num + i], 
            output_data[3 * out_num + i], 
            output_data[4 * out_num + i],
            output_data[5 * out_num + i + max_conf[i]],
            max_conf[i]));
    }
    
}



template <typename DeviceTensor>
class yolo_net {

public:

    yolo_net(const std::string &param_file) { 

        compile_opencl_kernels(conv_opencl_funcs, " ");
        
        load_weight_to_tensor(param_file, param);

    }

    void inference( const std::vector<float> &data_from_user, std::vector<float> &data_to_user) {
        
        std::vector<DetectedInfo> detected_result;


        DeviceTensor x(data_from_user);

        x = leaky_1(bn_1(conv_1(leaky_0(bn_0(conv_0(x))))));
        x += leaky_3(bn_3(conv_3(leaky_2(bn_2(conv_2(x))))));
        x = leaky_5(bn_5(conv_5(x)));
        x += leaky_7(bn_7(conv_7(leaky_6(bn_6(conv_6(x))))));
        x += leaky_10(bn_10(conv_10(leaky_9(bn_9(conv_9(x))))));
        x = leaky_12(bn_12(conv_12(x)));
        x += leaky_14(bn_14(conv_14(leaky_13(bn_13(conv_13(x))))));
        x += leaky_17(bn_17(conv_17(leaky_16(bn_16(conv_16(x))))));
        x += leaky_20(bn_20(conv_20(leaky_19(bn_19(conv_19(x))))));
        x += leaky_23(bn_23(conv_23(leaky_22(bn_22(conv_22(x))))));
        x += leaky_26(bn_26(conv_26(leaky_25(bn_25(conv_25(x))))));
        x += leaky_29(bn_29(conv_29(leaky_28(bn_28(conv_28(x))))));
        x += leaky_32(bn_32(conv_32(leaky_31(bn_31(conv_31(x))))));
        x += leaky_35(bn_35(conv_35(leaky_34(bn_34(conv_34(x)))))); auto x1 = x;
        x = leaky_37(bn_37(conv_37(x)));
        x += leaky_39(bn_39(conv_39(leaky_38(bn_38(conv_38(x))))));
        x += leaky_42(bn_42(conv_42(leaky_41(bn_41(conv_41(x))))));
        x += leaky_45(bn_45(conv_45(leaky_44(bn_44(conv_44(x))))));
        x += leaky_48(bn_48(conv_48(leaky_47(bn_47(conv_47(x))))));
        x += leaky_51(bn_51(conv_51(leaky_50(bn_50(conv_50(x))))));
        x += leaky_54(bn_54(conv_54(leaky_53(bn_53(conv_53(x))))));
        x += leaky_57(bn_57(conv_57(leaky_56(bn_56(conv_56(x))))));
        x += leaky_60(bn_60(conv_60(leaky_59(bn_59(conv_59(x)))))); auto x2 = x;
        x = leaky_62(bn_62(conv_62(x)));
        x += leaky_64(bn_64(conv_64(leaky_63(bn_63(conv_63(x))))));
        x += leaky_67(bn_67(conv_67(leaky_66(bn_66(conv_66(x))))));
        x += leaky_70(bn_70(conv_70(leaky_69(bn_69(conv_69(x))))));
        x += leaky_73(bn_73(conv_73(leaky_72(bn_72(conv_72(x))))));
        x = leaky_75(bn_75(conv_75(x)));
        x = leaky_76(bn_76(conv_76(x)));
        x = leaky_77(bn_77(conv_77(x)));
        x = leaky_78(bn_78(conv_78(x)));
        x = leaky_79(bn_79(conv_79(x)));
        


        predict_transform(
            conv_81(leaky_80(bn_80(conv_80(x)))), 
            1, 32, 13, 
            std::vector<float> {116, 90, 156, 198, 373, 326}, 
            80, 0.4, detected_result
        );




        x = leaky_84(bn_84(conv_84(x)));
        x = upsampling_85(x);

        x = concate(std::vector<DeviceTensor *> {&x, &x2});

        x = leaky_87(bn_87(conv_87(x)));
        x = leaky_88(bn_88(conv_88(x)));
        x = leaky_89(bn_89(conv_89(x)));
        x = leaky_90(bn_90(conv_90(x)));
        x = leaky_91(bn_91(conv_91(x)));


        predict_transform(
            conv_93(leaky_92(bn_92(conv_92(x)))), 
            1, 16, 26, 
            std::vector<float> {30, 61, 62, 45, 59, 119}, 
            80, 0.4, detected_result
        );


        x = leaky_96(bn_96(conv_96(x)));
        x = upsampling_97(x);
        x = concate(std::vector<DeviceTensor *> {&x, &x1}); //x1 + x;
        x = leaky_99(bn_99(conv_99(x)));
        x = leaky_100(bn_100(conv_100(x)));
        x = leaky_101(bn_101(conv_101(x)));
        x = leaky_102(bn_102(conv_102(x)));
        x = leaky_103(bn_103(conv_103(x)));


        predict_transform(
            conv_105(leaky_104(bn_104(conv_104(x)))), 
            1, 8, 52, 
            std::vector<float> {10, 13, 16, 30, 33, 23}, 
            80, 0.4, detected_result
        );

        

        for (int i = 0; i < detected_result.size(); ++i){



            std::cout << "This is "<< i << "-th " 
                     << detected_result[i].x1_ << " "
                        << detected_result[i].y1_ << " "
                        << detected_result[i].x2_ << " "
                        << detected_result[i].y2_ << " "
                        << detected_result[i].object_conf_ << " "
                        << detected_result[i].pos_conf_ << " "
                        << detected_result[i].object_index_ << std::endl;

        }

    }

private:
    
    DeviceTensor param = DeviceTensor(62001757);

     DeviceTensor conv_0_weight = param.sub_view(0, 864);
     DeviceTensor bn_0_mean = param.sub_view(864, 32);
     DeviceTensor bn_0_var = param.sub_view(896, 32);
     DeviceTensor bn_0_weight = param.sub_view(928, 32);
     DeviceTensor bn_0_bias = param.sub_view(960, 32);
     DeviceTensor conv_1_weight = param.sub_view(992, 18432);
     DeviceTensor bn_1_mean = param.sub_view(19424, 64);
     DeviceTensor bn_1_var = param.sub_view(19488, 64);
     DeviceTensor bn_1_weight = param.sub_view(19552, 64);
     DeviceTensor bn_1_bias = param.sub_view(19616, 64);
     DeviceTensor conv_2_weight = param.sub_view(19680, 2048);
     DeviceTensor bn_2_mean = param.sub_view(21728, 32);
     DeviceTensor bn_2_var = param.sub_view(21760, 32);
     DeviceTensor bn_2_weight = param.sub_view(21792, 32);
     DeviceTensor bn_2_bias = param.sub_view(21824, 32);
     DeviceTensor conv_3_weight = param.sub_view(21856, 18432);
     DeviceTensor bn_3_mean = param.sub_view(40288, 64);
     DeviceTensor bn_3_var = param.sub_view(40352, 64);
     DeviceTensor bn_3_weight = param.sub_view(40416, 64);
     DeviceTensor bn_3_bias = param.sub_view(40480, 64);
     DeviceTensor conv_5_weight = param.sub_view(40544, 73728);
     DeviceTensor bn_5_mean = param.sub_view(114272, 128);
     DeviceTensor bn_5_var = param.sub_view(114400, 128);
     DeviceTensor bn_5_weight = param.sub_view(114528, 128);
     DeviceTensor bn_5_bias = param.sub_view(114656, 128);
     DeviceTensor conv_6_weight = param.sub_view(114784, 8192);
     DeviceTensor bn_6_mean = param.sub_view(122976, 64);
     DeviceTensor bn_6_var = param.sub_view(123040, 64);
     DeviceTensor bn_6_weight = param.sub_view(123104, 64);
     DeviceTensor bn_6_bias = param.sub_view(123168, 64);
     DeviceTensor conv_7_weight = param.sub_view(123232, 73728);
     DeviceTensor bn_7_mean = param.sub_view(196960, 128);
     DeviceTensor bn_7_var = param.sub_view(197088, 128);
     DeviceTensor bn_7_weight = param.sub_view(197216, 128);
     DeviceTensor bn_7_bias = param.sub_view(197344, 128);
     DeviceTensor conv_9_weight = param.sub_view(197472, 8192);
     DeviceTensor bn_9_mean = param.sub_view(205664, 64);
     DeviceTensor bn_9_var = param.sub_view(205728, 64);
     DeviceTensor bn_9_weight = param.sub_view(205792, 64);
     DeviceTensor bn_9_bias = param.sub_view(205856, 64);
     DeviceTensor conv_10_weight = param.sub_view(205920, 73728);
     DeviceTensor bn_10_mean = param.sub_view(279648, 128);
     DeviceTensor bn_10_var = param.sub_view(279776, 128);
     DeviceTensor bn_10_weight = param.sub_view(279904, 128);
     DeviceTensor bn_10_bias = param.sub_view(280032, 128);
     DeviceTensor conv_12_weight = param.sub_view(280160, 294912);
     DeviceTensor bn_12_mean = param.sub_view(575072, 256);
     DeviceTensor bn_12_var = param.sub_view(575328, 256);
     DeviceTensor bn_12_weight = param.sub_view(575584, 256);
     DeviceTensor bn_12_bias = param.sub_view(575840, 256);
     DeviceTensor conv_13_weight = param.sub_view(576096, 32768);
     DeviceTensor bn_13_mean = param.sub_view(608864, 128);
     DeviceTensor bn_13_var = param.sub_view(608992, 128);
     DeviceTensor bn_13_weight = param.sub_view(609120, 128);
     DeviceTensor bn_13_bias = param.sub_view(609248, 128);
     DeviceTensor conv_14_weight = param.sub_view(609376, 294912);
     DeviceTensor bn_14_mean = param.sub_view(904288, 256);
     DeviceTensor bn_14_var = param.sub_view(904544, 256);
     DeviceTensor bn_14_weight = param.sub_view(904800, 256);
     DeviceTensor bn_14_bias = param.sub_view(905056, 256);
     DeviceTensor conv_16_weight = param.sub_view(905312, 32768);
     DeviceTensor bn_16_mean = param.sub_view(938080, 128);
     DeviceTensor bn_16_var = param.sub_view(938208, 128);
     DeviceTensor bn_16_weight = param.sub_view(938336, 128);
     DeviceTensor bn_16_bias = param.sub_view(938464, 128);
     DeviceTensor conv_17_weight = param.sub_view(938592, 294912);
     DeviceTensor bn_17_mean = param.sub_view(1233504, 256);
     DeviceTensor bn_17_var = param.sub_view(1233760, 256);
     DeviceTensor bn_17_weight = param.sub_view(1234016, 256);
     DeviceTensor bn_17_bias = param.sub_view(1234272, 256);
     DeviceTensor conv_19_weight = param.sub_view(1234528, 32768);
     DeviceTensor bn_19_mean = param.sub_view(1267296, 128);
     DeviceTensor bn_19_var = param.sub_view(1267424, 128);
     DeviceTensor bn_19_weight = param.sub_view(1267552, 128);
     DeviceTensor bn_19_bias = param.sub_view(1267680, 128);
     DeviceTensor conv_20_weight = param.sub_view(1267808, 294912);
     DeviceTensor bn_20_mean = param.sub_view(1562720, 256);
     DeviceTensor bn_20_var = param.sub_view(1562976, 256);
     DeviceTensor bn_20_weight = param.sub_view(1563232, 256);
     DeviceTensor bn_20_bias = param.sub_view(1563488, 256);
     DeviceTensor conv_22_weight = param.sub_view(1563744, 32768);
     DeviceTensor bn_22_mean = param.sub_view(1596512, 128);
     DeviceTensor bn_22_var = param.sub_view(1596640, 128);
     DeviceTensor bn_22_weight = param.sub_view(1596768, 128);
     DeviceTensor bn_22_bias = param.sub_view(1596896, 128);
     DeviceTensor conv_23_weight = param.sub_view(1597024, 294912);
     DeviceTensor bn_23_mean = param.sub_view(1891936, 256);
     DeviceTensor bn_23_var = param.sub_view(1892192, 256);
     DeviceTensor bn_23_weight = param.sub_view(1892448, 256);
     DeviceTensor bn_23_bias = param.sub_view(1892704, 256);
     DeviceTensor conv_25_weight = param.sub_view(1892960, 32768);
     DeviceTensor bn_25_mean = param.sub_view(1925728, 128);
     DeviceTensor bn_25_var = param.sub_view(1925856, 128);
     DeviceTensor bn_25_weight = param.sub_view(1925984, 128);
     DeviceTensor bn_25_bias = param.sub_view(1926112, 128);
     DeviceTensor conv_26_weight = param.sub_view(1926240, 294912);
     DeviceTensor bn_26_mean = param.sub_view(2221152, 256);
     DeviceTensor bn_26_var = param.sub_view(2221408, 256);
     DeviceTensor bn_26_weight = param.sub_view(2221664, 256);
     DeviceTensor bn_26_bias = param.sub_view(2221920, 256);
     DeviceTensor conv_28_weight = param.sub_view(2222176, 32768);
     DeviceTensor bn_28_mean = param.sub_view(2254944, 128);
     DeviceTensor bn_28_var = param.sub_view(2255072, 128);
     DeviceTensor bn_28_weight = param.sub_view(2255200, 128);
     DeviceTensor bn_28_bias = param.sub_view(2255328, 128);
     DeviceTensor conv_29_weight = param.sub_view(2255456, 294912);
     DeviceTensor bn_29_mean = param.sub_view(2550368, 256);
     DeviceTensor bn_29_var = param.sub_view(2550624, 256);
     DeviceTensor bn_29_weight = param.sub_view(2550880, 256);
     DeviceTensor bn_29_bias = param.sub_view(2551136, 256);
     DeviceTensor conv_31_weight = param.sub_view(2551392, 32768);
     DeviceTensor bn_31_mean = param.sub_view(2584160, 128);
     DeviceTensor bn_31_var = param.sub_view(2584288, 128);
     DeviceTensor bn_31_weight = param.sub_view(2584416, 128);
     DeviceTensor bn_31_bias = param.sub_view(2584544, 128);
     DeviceTensor conv_32_weight = param.sub_view(2584672, 294912);
     DeviceTensor bn_32_mean = param.sub_view(2879584, 256);
     DeviceTensor bn_32_var = param.sub_view(2879840, 256);
     DeviceTensor bn_32_weight = param.sub_view(2880096, 256);
     DeviceTensor bn_32_bias = param.sub_view(2880352, 256);
     DeviceTensor conv_34_weight = param.sub_view(2880608, 32768);
     DeviceTensor bn_34_mean = param.sub_view(2913376, 128);
     DeviceTensor bn_34_var = param.sub_view(2913504, 128);
     DeviceTensor bn_34_weight = param.sub_view(2913632, 128);
     DeviceTensor bn_34_bias = param.sub_view(2913760, 128);
     DeviceTensor conv_35_weight = param.sub_view(2913888, 294912);
     DeviceTensor bn_35_mean = param.sub_view(3208800, 256);
     DeviceTensor bn_35_var = param.sub_view(3209056, 256);
     DeviceTensor bn_35_weight = param.sub_view(3209312, 256);
     DeviceTensor bn_35_bias = param.sub_view(3209568, 256);
     DeviceTensor conv_37_weight = param.sub_view(3209824, 1179648);
     DeviceTensor bn_37_mean = param.sub_view(4389472, 512);
     DeviceTensor bn_37_var = param.sub_view(4389984, 512);
     DeviceTensor bn_37_weight = param.sub_view(4390496, 512);
     DeviceTensor bn_37_bias = param.sub_view(4391008, 512);
     DeviceTensor conv_38_weight = param.sub_view(4391520, 131072);
     DeviceTensor bn_38_mean = param.sub_view(4522592, 256);
     DeviceTensor bn_38_var = param.sub_view(4522848, 256);
     DeviceTensor bn_38_weight = param.sub_view(4523104, 256);
     DeviceTensor bn_38_bias = param.sub_view(4523360, 256);
     DeviceTensor conv_39_weight = param.sub_view(4523616, 1179648);
     DeviceTensor bn_39_mean = param.sub_view(5703264, 512);
     DeviceTensor bn_39_var = param.sub_view(5703776, 512);
     DeviceTensor bn_39_weight = param.sub_view(5704288, 512);
     DeviceTensor bn_39_bias = param.sub_view(5704800, 512);
     DeviceTensor conv_41_weight = param.sub_view(5705312, 131072);
     DeviceTensor bn_41_mean = param.sub_view(5836384, 256);
     DeviceTensor bn_41_var = param.sub_view(5836640, 256);
     DeviceTensor bn_41_weight = param.sub_view(5836896, 256);
     DeviceTensor bn_41_bias = param.sub_view(5837152, 256);
     DeviceTensor conv_42_weight = param.sub_view(5837408, 1179648);
     DeviceTensor bn_42_mean = param.sub_view(7017056, 512);
     DeviceTensor bn_42_var = param.sub_view(7017568, 512);
     DeviceTensor bn_42_weight = param.sub_view(7018080, 512);
     DeviceTensor bn_42_bias = param.sub_view(7018592, 512);
     DeviceTensor conv_44_weight = param.sub_view(7019104, 131072);
     DeviceTensor bn_44_mean = param.sub_view(7150176, 256);
     DeviceTensor bn_44_var = param.sub_view(7150432, 256);
     DeviceTensor bn_44_weight = param.sub_view(7150688, 256);
     DeviceTensor bn_44_bias = param.sub_view(7150944, 256);
     DeviceTensor conv_45_weight = param.sub_view(7151200, 1179648);
     DeviceTensor bn_45_mean = param.sub_view(8330848, 512);
     DeviceTensor bn_45_var = param.sub_view(8331360, 512);
     DeviceTensor bn_45_weight = param.sub_view(8331872, 512);
     DeviceTensor bn_45_bias = param.sub_view(8332384, 512);
     DeviceTensor conv_47_weight = param.sub_view(8332896, 131072);
     DeviceTensor bn_47_mean = param.sub_view(8463968, 256);
     DeviceTensor bn_47_var = param.sub_view(8464224, 256);
     DeviceTensor bn_47_weight = param.sub_view(8464480, 256);
     DeviceTensor bn_47_bias = param.sub_view(8464736, 256);
     DeviceTensor conv_48_weight = param.sub_view(8464992, 1179648);
     DeviceTensor bn_48_mean = param.sub_view(9644640, 512);
     DeviceTensor bn_48_var = param.sub_view(9645152, 512);
     DeviceTensor bn_48_weight = param.sub_view(9645664, 512);
     DeviceTensor bn_48_bias = param.sub_view(9646176, 512);
     DeviceTensor conv_50_weight = param.sub_view(9646688, 131072);
     DeviceTensor bn_50_mean = param.sub_view(9777760, 256);
     DeviceTensor bn_50_var = param.sub_view(9778016, 256);
     DeviceTensor bn_50_weight = param.sub_view(9778272, 256);
     DeviceTensor bn_50_bias = param.sub_view(9778528, 256);
     DeviceTensor conv_51_weight = param.sub_view(9778784, 1179648);
     DeviceTensor bn_51_mean = param.sub_view(10958432, 512);
     DeviceTensor bn_51_var = param.sub_view(10958944, 512);
     DeviceTensor bn_51_weight = param.sub_view(10959456, 512);
     DeviceTensor bn_51_bias = param.sub_view(10959968, 512);
     DeviceTensor conv_53_weight = param.sub_view(10960480, 131072);
     DeviceTensor bn_53_mean = param.sub_view(11091552, 256);
     DeviceTensor bn_53_var = param.sub_view(11091808, 256);
     DeviceTensor bn_53_weight = param.sub_view(11092064, 256);
     DeviceTensor bn_53_bias = param.sub_view(11092320, 256);
     DeviceTensor conv_54_weight = param.sub_view(11092576, 1179648);
     DeviceTensor bn_54_mean = param.sub_view(12272224, 512);
     DeviceTensor bn_54_var = param.sub_view(12272736, 512);
     DeviceTensor bn_54_weight = param.sub_view(12273248, 512);
     DeviceTensor bn_54_bias = param.sub_view(12273760, 512);
     DeviceTensor conv_56_weight = param.sub_view(12274272, 131072);
     DeviceTensor bn_56_mean = param.sub_view(12405344, 256);
     DeviceTensor bn_56_var = param.sub_view(12405600, 256);
     DeviceTensor bn_56_weight = param.sub_view(12405856, 256);
     DeviceTensor bn_56_bias = param.sub_view(12406112, 256);
     DeviceTensor conv_57_weight = param.sub_view(12406368, 1179648);
     DeviceTensor bn_57_mean = param.sub_view(13586016, 512);
     DeviceTensor bn_57_var = param.sub_view(13586528, 512);
     DeviceTensor bn_57_weight = param.sub_view(13587040, 512);
     DeviceTensor bn_57_bias = param.sub_view(13587552, 512);
     DeviceTensor conv_59_weight = param.sub_view(13588064, 131072);
     DeviceTensor bn_59_mean = param.sub_view(13719136, 256);
     DeviceTensor bn_59_var = param.sub_view(13719392, 256);
     DeviceTensor bn_59_weight = param.sub_view(13719648, 256);
     DeviceTensor bn_59_bias = param.sub_view(13719904, 256);
     DeviceTensor conv_60_weight = param.sub_view(13720160, 1179648);
     DeviceTensor bn_60_mean = param.sub_view(14899808, 512);
     DeviceTensor bn_60_var = param.sub_view(14900320, 512);
     DeviceTensor bn_60_weight = param.sub_view(14900832, 512);
     DeviceTensor bn_60_bias = param.sub_view(14901344, 512);
     DeviceTensor conv_62_weight = param.sub_view(14901856, 4718592);
     DeviceTensor bn_62_mean = param.sub_view(19620448, 1024);
     DeviceTensor bn_62_var = param.sub_view(19621472, 1024);
     DeviceTensor bn_62_weight = param.sub_view(19622496, 1024);
     DeviceTensor bn_62_bias = param.sub_view(19623520, 1024);
     DeviceTensor conv_63_weight = param.sub_view(19624544, 524288);
     DeviceTensor bn_63_mean = param.sub_view(20148832, 512);
     DeviceTensor bn_63_var = param.sub_view(20149344, 512);
     DeviceTensor bn_63_weight = param.sub_view(20149856, 512);
     DeviceTensor bn_63_bias = param.sub_view(20150368, 512);
     DeviceTensor conv_64_weight = param.sub_view(20150880, 4718592);
     DeviceTensor bn_64_mean = param.sub_view(24869472, 1024);
     DeviceTensor bn_64_var = param.sub_view(24870496, 1024);
     DeviceTensor bn_64_weight = param.sub_view(24871520, 1024);
     DeviceTensor bn_64_bias = param.sub_view(24872544, 1024);
     DeviceTensor conv_66_weight = param.sub_view(24873568, 524288);
     DeviceTensor bn_66_mean = param.sub_view(25397856, 512);
     DeviceTensor bn_66_var = param.sub_view(25398368, 512);
     DeviceTensor bn_66_weight = param.sub_view(25398880, 512);
     DeviceTensor bn_66_bias = param.sub_view(25399392, 512);
     DeviceTensor conv_67_weight = param.sub_view(25399904, 4718592);
     DeviceTensor bn_67_mean = param.sub_view(30118496, 1024);
     DeviceTensor bn_67_var = param.sub_view(30119520, 1024);
     DeviceTensor bn_67_weight = param.sub_view(30120544, 1024);
     DeviceTensor bn_67_bias = param.sub_view(30121568, 1024);
     DeviceTensor conv_69_weight = param.sub_view(30122592, 524288);
     DeviceTensor bn_69_mean = param.sub_view(30646880, 512);
     DeviceTensor bn_69_var = param.sub_view(30647392, 512);
     DeviceTensor bn_69_weight = param.sub_view(30647904, 512);
     DeviceTensor bn_69_bias = param.sub_view(30648416, 512);
     DeviceTensor conv_70_weight = param.sub_view(30648928, 4718592);
     DeviceTensor bn_70_mean = param.sub_view(35367520, 1024);
     DeviceTensor bn_70_var = param.sub_view(35368544, 1024);
     DeviceTensor bn_70_weight = param.sub_view(35369568, 1024);
     DeviceTensor bn_70_bias = param.sub_view(35370592, 1024);
     DeviceTensor conv_72_weight = param.sub_view(35371616, 524288);
     DeviceTensor bn_72_mean = param.sub_view(35895904, 512);
     DeviceTensor bn_72_var = param.sub_view(35896416, 512);
     DeviceTensor bn_72_weight = param.sub_view(35896928, 512);
     DeviceTensor bn_72_bias = param.sub_view(35897440, 512);
     DeviceTensor conv_73_weight = param.sub_view(35897952, 4718592);
     DeviceTensor bn_73_mean = param.sub_view(40616544, 1024);
     DeviceTensor bn_73_var = param.sub_view(40617568, 1024);
     DeviceTensor bn_73_weight = param.sub_view(40618592, 1024);
     DeviceTensor bn_73_bias = param.sub_view(40619616, 1024);
     DeviceTensor conv_75_weight = param.sub_view(40620640, 524288);
     DeviceTensor bn_75_mean = param.sub_view(41144928, 512);
     DeviceTensor bn_75_var = param.sub_view(41145440, 512);
     DeviceTensor bn_75_weight = param.sub_view(41145952, 512);
     DeviceTensor bn_75_bias = param.sub_view(41146464, 512);
     DeviceTensor conv_76_weight = param.sub_view(41146976, 4718592);
     DeviceTensor bn_76_mean = param.sub_view(45865568, 1024);
     DeviceTensor bn_76_var = param.sub_view(45866592, 1024);
     DeviceTensor bn_76_weight = param.sub_view(45867616, 1024);
     DeviceTensor bn_76_bias = param.sub_view(45868640, 1024);
     DeviceTensor conv_77_weight = param.sub_view(45869664, 524288);
     DeviceTensor bn_77_mean = param.sub_view(46393952, 512);
     DeviceTensor bn_77_var = param.sub_view(46394464, 512);
     DeviceTensor bn_77_weight = param.sub_view(46394976, 512);
     DeviceTensor bn_77_bias = param.sub_view(46395488, 512);
     DeviceTensor conv_78_weight = param.sub_view(46396000, 4718592);
     DeviceTensor bn_78_mean = param.sub_view(51114592, 1024);
     DeviceTensor bn_78_var = param.sub_view(51115616, 1024);
     DeviceTensor bn_78_weight = param.sub_view(51116640, 1024);
     DeviceTensor bn_78_bias = param.sub_view(51117664, 1024);
     DeviceTensor conv_79_weight = param.sub_view(51118688, 524288);
     DeviceTensor bn_79_mean = param.sub_view(51642976, 512);
     DeviceTensor bn_79_var = param.sub_view(51643488, 512);
     DeviceTensor bn_79_weight = param.sub_view(51644000, 512);
     DeviceTensor bn_79_bias = param.sub_view(51644512, 512);
     DeviceTensor conv_80_weight = param.sub_view(51645024, 4718592);
     DeviceTensor bn_80_mean = param.sub_view(56363616, 1024);
     DeviceTensor bn_80_var = param.sub_view(56364640, 1024);
     DeviceTensor bn_80_weight = param.sub_view(56365664, 1024);
     DeviceTensor bn_80_bias = param.sub_view(56366688, 1024);
     DeviceTensor conv_81_bias = param.sub_view(56367712, 255);
     DeviceTensor conv_81_weight = param.sub_view(56367967, 261120);
     DeviceTensor conv_84_weight = param.sub_view(56629087, 131072);
     DeviceTensor bn_84_mean = param.sub_view(56760159, 256);
     DeviceTensor bn_84_var = param.sub_view(56760415, 256);
     DeviceTensor bn_84_weight = param.sub_view(56760671, 256);
     DeviceTensor bn_84_bias = param.sub_view(56760927, 256);
     DeviceTensor conv_87_weight = param.sub_view(56761183, 196608);
     DeviceTensor bn_87_mean = param.sub_view(56957791, 256);
     DeviceTensor bn_87_var = param.sub_view(56958047, 256);
     DeviceTensor bn_87_weight = param.sub_view(56958303, 256);
     DeviceTensor bn_87_bias = param.sub_view(56958559, 256);
     DeviceTensor conv_88_weight = param.sub_view(56958815, 1179648);
     DeviceTensor bn_88_mean = param.sub_view(58138463, 512);
     DeviceTensor bn_88_var = param.sub_view(58138975, 512);
     DeviceTensor bn_88_weight = param.sub_view(58139487, 512);
     DeviceTensor bn_88_bias = param.sub_view(58139999, 512);
     DeviceTensor conv_89_weight = param.sub_view(58140511, 131072);
     DeviceTensor bn_89_mean = param.sub_view(58271583, 256);
     DeviceTensor bn_89_var = param.sub_view(58271839, 256);
     DeviceTensor bn_89_weight = param.sub_view(58272095, 256);
     DeviceTensor bn_89_bias = param.sub_view(58272351, 256);
     DeviceTensor conv_90_weight = param.sub_view(58272607, 1179648);
     DeviceTensor bn_90_mean = param.sub_view(59452255, 512);
     DeviceTensor bn_90_var = param.sub_view(59452767, 512);
     DeviceTensor bn_90_weight = param.sub_view(59453279, 512);
     DeviceTensor bn_90_bias = param.sub_view(59453791, 512);
     DeviceTensor conv_91_weight = param.sub_view(59454303, 131072);
     DeviceTensor bn_91_mean = param.sub_view(59585375, 256);
     DeviceTensor bn_91_var = param.sub_view(59585631, 256);
     DeviceTensor bn_91_weight = param.sub_view(59585887, 256);
     DeviceTensor bn_91_bias = param.sub_view(59586143, 256);
     DeviceTensor conv_92_weight = param.sub_view(59586399, 1179648);
     DeviceTensor bn_92_mean = param.sub_view(60766047, 512);
     DeviceTensor bn_92_var = param.sub_view(60766559, 512);
     DeviceTensor bn_92_weight = param.sub_view(60767071, 512);
     DeviceTensor bn_92_bias = param.sub_view(60767583, 512);
     DeviceTensor conv_93_bias = param.sub_view(60768095, 255);
     DeviceTensor conv_93_weight = param.sub_view(60768350, 130560);
     DeviceTensor conv_96_weight = param.sub_view(60898910, 32768);
     DeviceTensor bn_96_mean = param.sub_view(60931678, 128);
     DeviceTensor bn_96_var = param.sub_view(60931806, 128);
     DeviceTensor bn_96_weight = param.sub_view(60931934, 128);
     DeviceTensor bn_96_bias = param.sub_view(60932062, 128);
     DeviceTensor conv_99_weight = param.sub_view(60932190, 49152);
     DeviceTensor bn_99_mean = param.sub_view(60981342, 128);
     DeviceTensor bn_99_var = param.sub_view(60981470, 128);
     DeviceTensor bn_99_weight = param.sub_view(60981598, 128);
     DeviceTensor bn_99_bias = param.sub_view(60981726, 128);
     DeviceTensor conv_100_weight = param.sub_view(60981854, 294912);
     DeviceTensor bn_100_mean = param.sub_view(61276766, 256);
     DeviceTensor bn_100_var = param.sub_view(61277022, 256);
     DeviceTensor bn_100_weight = param.sub_view(61277278, 256);
     DeviceTensor bn_100_bias = param.sub_view(61277534, 256);
     DeviceTensor conv_101_weight = param.sub_view(61277790, 32768);
     DeviceTensor bn_101_mean = param.sub_view(61310558, 128);
     DeviceTensor bn_101_var = param.sub_view(61310686, 128);
     DeviceTensor bn_101_weight = param.sub_view(61310814, 128);
     DeviceTensor bn_101_bias = param.sub_view(61310942, 128);
     DeviceTensor conv_102_weight = param.sub_view(61311070, 294912);
     DeviceTensor bn_102_mean = param.sub_view(61605982, 256);
     DeviceTensor bn_102_var = param.sub_view(61606238, 256);
     DeviceTensor bn_102_weight = param.sub_view(61606494, 256);
     DeviceTensor bn_102_bias = param.sub_view(61606750, 256);
     DeviceTensor conv_103_weight = param.sub_view(61607006, 32768);
     DeviceTensor bn_103_mean = param.sub_view(61639774, 128);
     DeviceTensor bn_103_var = param.sub_view(61639902, 128);
     DeviceTensor bn_103_weight = param.sub_view(61640030, 128);
     DeviceTensor bn_103_bias = param.sub_view(61640158, 128);
     DeviceTensor conv_104_weight = param.sub_view(61640286, 294912);
     DeviceTensor bn_104_mean = param.sub_view(61935198, 256);
     DeviceTensor bn_104_var = param.sub_view(61935454, 256);
     DeviceTensor bn_104_weight = param.sub_view(61935710, 256);
     DeviceTensor bn_104_bias = param.sub_view(61935966, 256);
     DeviceTensor conv_105_bias = param.sub_view(61936222, 255);
     DeviceTensor conv_105_weight = param.sub_view(61936477, 65280);
    LibDNNConvOp<DeviceTensor> conv_0 = LibDNNConvOp<DeviceTensor> ("conv_0_forward", 5537792, &conv_0_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {21632,8,1});
    BatchNormOp<DeviceTensor> bn_0 = BatchNormOp<DeviceTensor> (32, 173056, 1e-05, &bn_0_mean, &bn_0_var, &bn_0_weight, &bn_0_bias);
    ReLUOp<DeviceTensor> leaky_0 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_1 = LibDNNConvOp<DeviceTensor> ("conv_1_forward", 2768896, &conv_1_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {5408,16,1});
    BatchNormOp<DeviceTensor> bn_1 = BatchNormOp<DeviceTensor> (64, 43264, 1e-05, &bn_1_mean, &bn_1_var, &bn_1_weight, &bn_1_bias);
    ReLUOp<DeviceTensor> leaky_1 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_2 = LibDNNConvOp<DeviceTensor> ("conv_2_forward", 1384448, &conv_2_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {5408,8,1});
    BatchNormOp<DeviceTensor> bn_2 = BatchNormOp<DeviceTensor> (32, 43264, 1e-05, &bn_2_mean, &bn_2_var, &bn_2_weight, &bn_2_bias);
    ReLUOp<DeviceTensor> leaky_2 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_3 = LibDNNConvOp<DeviceTensor> ("conv_3_forward", 2768896, &conv_3_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {5408,16,1});
    BatchNormOp<DeviceTensor> bn_3 = BatchNormOp<DeviceTensor> (64, 43264, 1e-05, &bn_3_mean, &bn_3_var, &bn_3_weight, &bn_3_bias);
    ReLUOp<DeviceTensor> leaky_3 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_5 = LibDNNConvOp<DeviceTensor> ("conv_5_forward", 1384448, &conv_5_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {1360,32,1});
    BatchNormOp<DeviceTensor> bn_5 = BatchNormOp<DeviceTensor> (128, 10816, 1e-05, &bn_5_mean, &bn_5_var, &bn_5_weight, &bn_5_bias);
    ReLUOp<DeviceTensor> leaky_5 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_6 = LibDNNConvOp<DeviceTensor> ("conv_6_forward", 692224, &conv_6_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {1360,16,1});
    BatchNormOp<DeviceTensor> bn_6 = BatchNormOp<DeviceTensor> (64, 10816, 1e-05, &bn_6_mean, &bn_6_var, &bn_6_weight, &bn_6_bias);
    ReLUOp<DeviceTensor> leaky_6 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_7 = LibDNNConvOp<DeviceTensor> ("conv_7_forward", 1384448, &conv_7_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {1360,32,1});
    BatchNormOp<DeviceTensor> bn_7 = BatchNormOp<DeviceTensor> (128, 10816, 1e-05, &bn_7_mean, &bn_7_var, &bn_7_weight, &bn_7_bias);
    ReLUOp<DeviceTensor> leaky_7 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_9 = LibDNNConvOp<DeviceTensor> ("conv_9_forward", 692224, &conv_9_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {1360,16,1});
    BatchNormOp<DeviceTensor> bn_9 = BatchNormOp<DeviceTensor> (64, 10816, 1e-05, &bn_9_mean, &bn_9_var, &bn_9_weight, &bn_9_bias);
    ReLUOp<DeviceTensor> leaky_9 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_10 = LibDNNConvOp<DeviceTensor> ("conv_10_forward", 1384448, &conv_10_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {1360,32,1});
    BatchNormOp<DeviceTensor> bn_10 = BatchNormOp<DeviceTensor> (128, 10816, 1e-05, &bn_10_mean, &bn_10_var, &bn_10_weight, &bn_10_bias);
    ReLUOp<DeviceTensor> leaky_10 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_12 = LibDNNConvOp<DeviceTensor> ("conv_12_forward", 692224, &conv_12_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> bn_12 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &bn_12_mean, &bn_12_var, &bn_12_weight, &bn_12_bias);
    ReLUOp<DeviceTensor> leaky_12 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_13 = LibDNNConvOp<DeviceTensor> ("conv_13_forward", 346112, &conv_13_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,32,1});
    BatchNormOp<DeviceTensor> bn_13 = BatchNormOp<DeviceTensor> (128, 2704, 1e-05, &bn_13_mean, &bn_13_var, &bn_13_weight, &bn_13_bias);
    ReLUOp<DeviceTensor> leaky_13 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_14 = LibDNNConvOp<DeviceTensor> ("conv_14_forward", 692224, &conv_14_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> bn_14 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &bn_14_mean, &bn_14_var, &bn_14_weight, &bn_14_bias);
    ReLUOp<DeviceTensor> leaky_14 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_16 = LibDNNConvOp<DeviceTensor> ("conv_16_forward", 346112, &conv_16_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,32,1});
    BatchNormOp<DeviceTensor> bn_16 = BatchNormOp<DeviceTensor> (128, 2704, 1e-05, &bn_16_mean, &bn_16_var, &bn_16_weight, &bn_16_bias);
    ReLUOp<DeviceTensor> leaky_16 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_17 = LibDNNConvOp<DeviceTensor> ("conv_17_forward", 692224, &conv_17_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> bn_17 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &bn_17_mean, &bn_17_var, &bn_17_weight, &bn_17_bias);
    ReLUOp<DeviceTensor> leaky_17 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_19 = LibDNNConvOp<DeviceTensor> ("conv_19_forward", 346112, &conv_19_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,32,1});
    BatchNormOp<DeviceTensor> bn_19 = BatchNormOp<DeviceTensor> (128, 2704, 1e-05, &bn_19_mean, &bn_19_var, &bn_19_weight, &bn_19_bias);
    ReLUOp<DeviceTensor> leaky_19 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_20 = LibDNNConvOp<DeviceTensor> ("conv_20_forward", 692224, &conv_20_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> bn_20 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &bn_20_mean, &bn_20_var, &bn_20_weight, &bn_20_bias);
    ReLUOp<DeviceTensor> leaky_20 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_22 = LibDNNConvOp<DeviceTensor> ("conv_22_forward", 346112, &conv_22_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,32,1});
    BatchNormOp<DeviceTensor> bn_22 = BatchNormOp<DeviceTensor> (128, 2704, 1e-05, &bn_22_mean, &bn_22_var, &bn_22_weight, &bn_22_bias);
    ReLUOp<DeviceTensor> leaky_22 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_23 = LibDNNConvOp<DeviceTensor> ("conv_23_forward", 692224, &conv_23_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> bn_23 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &bn_23_mean, &bn_23_var, &bn_23_weight, &bn_23_bias);
    ReLUOp<DeviceTensor> leaky_23 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_25 = LibDNNConvOp<DeviceTensor> ("conv_25_forward", 346112, &conv_25_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,32,1});
    BatchNormOp<DeviceTensor> bn_25 = BatchNormOp<DeviceTensor> (128, 2704, 1e-05, &bn_25_mean, &bn_25_var, &bn_25_weight, &bn_25_bias);
    ReLUOp<DeviceTensor> leaky_25 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_26 = LibDNNConvOp<DeviceTensor> ("conv_26_forward", 692224, &conv_26_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> bn_26 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &bn_26_mean, &bn_26_var, &bn_26_weight, &bn_26_bias);
    ReLUOp<DeviceTensor> leaky_26 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_28 = LibDNNConvOp<DeviceTensor> ("conv_28_forward", 346112, &conv_28_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,32,1});
    BatchNormOp<DeviceTensor> bn_28 = BatchNormOp<DeviceTensor> (128, 2704, 1e-05, &bn_28_mean, &bn_28_var, &bn_28_weight, &bn_28_bias);
    ReLUOp<DeviceTensor> leaky_28 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_29 = LibDNNConvOp<DeviceTensor> ("conv_29_forward", 692224, &conv_29_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> bn_29 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &bn_29_mean, &bn_29_var, &bn_29_weight, &bn_29_bias);
    ReLUOp<DeviceTensor> leaky_29 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_31 = LibDNNConvOp<DeviceTensor> ("conv_31_forward", 346112, &conv_31_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,32,1});
    BatchNormOp<DeviceTensor> bn_31 = BatchNormOp<DeviceTensor> (128, 2704, 1e-05, &bn_31_mean, &bn_31_var, &bn_31_weight, &bn_31_bias);
    ReLUOp<DeviceTensor> leaky_31 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_32 = LibDNNConvOp<DeviceTensor> ("conv_32_forward", 692224, &conv_32_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> bn_32 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &bn_32_mean, &bn_32_var, &bn_32_weight, &bn_32_bias);
    ReLUOp<DeviceTensor> leaky_32 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_34 = LibDNNConvOp<DeviceTensor> ("conv_34_forward", 346112, &conv_34_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,32,1});
    BatchNormOp<DeviceTensor> bn_34 = BatchNormOp<DeviceTensor> (128, 2704, 1e-05, &bn_34_mean, &bn_34_var, &bn_34_weight, &bn_34_bias);
    ReLUOp<DeviceTensor> leaky_34 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_35 = LibDNNConvOp<DeviceTensor> ("conv_35_forward", 692224, &conv_35_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> bn_35 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &bn_35_mean, &bn_35_var, &bn_35_weight, &bn_35_bias);
    ReLUOp<DeviceTensor> leaky_35 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_37 = LibDNNConvOp<DeviceTensor> ("conv_37_forward", 346112, &conv_37_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> bn_37 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &bn_37_mean, &bn_37_var, &bn_37_weight, &bn_37_bias);
    ReLUOp<DeviceTensor> leaky_37 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_38 = LibDNNConvOp<DeviceTensor> ("conv_38_forward", 173056, &conv_38_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    BatchNormOp<DeviceTensor> bn_38 = BatchNormOp<DeviceTensor> (256, 676, 1e-05, &bn_38_mean, &bn_38_var, &bn_38_weight, &bn_38_bias);
    ReLUOp<DeviceTensor> leaky_38 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_39 = LibDNNConvOp<DeviceTensor> ("conv_39_forward", 346112, &conv_39_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> bn_39 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &bn_39_mean, &bn_39_var, &bn_39_weight, &bn_39_bias);
    ReLUOp<DeviceTensor> leaky_39 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_41 = LibDNNConvOp<DeviceTensor> ("conv_41_forward", 173056, &conv_41_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    BatchNormOp<DeviceTensor> bn_41 = BatchNormOp<DeviceTensor> (256, 676, 1e-05, &bn_41_mean, &bn_41_var, &bn_41_weight, &bn_41_bias);
    ReLUOp<DeviceTensor> leaky_41 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_42 = LibDNNConvOp<DeviceTensor> ("conv_42_forward", 346112, &conv_42_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> bn_42 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &bn_42_mean, &bn_42_var, &bn_42_weight, &bn_42_bias);
    ReLUOp<DeviceTensor> leaky_42 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_44 = LibDNNConvOp<DeviceTensor> ("conv_44_forward", 173056, &conv_44_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    BatchNormOp<DeviceTensor> bn_44 = BatchNormOp<DeviceTensor> (256, 676, 1e-05, &bn_44_mean, &bn_44_var, &bn_44_weight, &bn_44_bias);
    ReLUOp<DeviceTensor> leaky_44 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_45 = LibDNNConvOp<DeviceTensor> ("conv_45_forward", 346112, &conv_45_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> bn_45 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &bn_45_mean, &bn_45_var, &bn_45_weight, &bn_45_bias);
    ReLUOp<DeviceTensor> leaky_45 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_47 = LibDNNConvOp<DeviceTensor> ("conv_47_forward", 173056, &conv_47_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    BatchNormOp<DeviceTensor> bn_47 = BatchNormOp<DeviceTensor> (256, 676, 1e-05, &bn_47_mean, &bn_47_var, &bn_47_weight, &bn_47_bias);
    ReLUOp<DeviceTensor> leaky_47 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_48 = LibDNNConvOp<DeviceTensor> ("conv_48_forward", 346112, &conv_48_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> bn_48 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &bn_48_mean, &bn_48_var, &bn_48_weight, &bn_48_bias);
    ReLUOp<DeviceTensor> leaky_48 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_50 = LibDNNConvOp<DeviceTensor> ("conv_50_forward", 173056, &conv_50_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    BatchNormOp<DeviceTensor> bn_50 = BatchNormOp<DeviceTensor> (256, 676, 1e-05, &bn_50_mean, &bn_50_var, &bn_50_weight, &bn_50_bias);
    ReLUOp<DeviceTensor> leaky_50 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_51 = LibDNNConvOp<DeviceTensor> ("conv_51_forward", 346112, &conv_51_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> bn_51 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &bn_51_mean, &bn_51_var, &bn_51_weight, &bn_51_bias);
    ReLUOp<DeviceTensor> leaky_51 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_53 = LibDNNConvOp<DeviceTensor> ("conv_53_forward", 173056, &conv_53_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    BatchNormOp<DeviceTensor> bn_53 = BatchNormOp<DeviceTensor> (256, 676, 1e-05, &bn_53_mean, &bn_53_var, &bn_53_weight, &bn_53_bias);
    ReLUOp<DeviceTensor> leaky_53 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_54 = LibDNNConvOp<DeviceTensor> ("conv_54_forward", 346112, &conv_54_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> bn_54 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &bn_54_mean, &bn_54_var, &bn_54_weight, &bn_54_bias);
    ReLUOp<DeviceTensor> leaky_54 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_56 = LibDNNConvOp<DeviceTensor> ("conv_56_forward", 173056, &conv_56_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    BatchNormOp<DeviceTensor> bn_56 = BatchNormOp<DeviceTensor> (256, 676, 1e-05, &bn_56_mean, &bn_56_var, &bn_56_weight, &bn_56_bias);
    ReLUOp<DeviceTensor> leaky_56 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_57 = LibDNNConvOp<DeviceTensor> ("conv_57_forward", 346112, &conv_57_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> bn_57 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &bn_57_mean, &bn_57_var, &bn_57_weight, &bn_57_bias);
    ReLUOp<DeviceTensor> leaky_57 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_59 = LibDNNConvOp<DeviceTensor> ("conv_59_forward", 173056, &conv_59_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    BatchNormOp<DeviceTensor> bn_59 = BatchNormOp<DeviceTensor> (256, 676, 1e-05, &bn_59_mean, &bn_59_var, &bn_59_weight, &bn_59_bias);
    ReLUOp<DeviceTensor> leaky_59 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_60 = LibDNNConvOp<DeviceTensor> ("conv_60_forward", 346112, &conv_60_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> bn_60 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &bn_60_mean, &bn_60_var, &bn_60_weight, &bn_60_bias);
    ReLUOp<DeviceTensor> leaky_60 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_62 = LibDNNConvOp<DeviceTensor> ("conv_62_forward", 173056, &conv_62_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,256,1});
    BatchNormOp<DeviceTensor> bn_62 = BatchNormOp<DeviceTensor> (1024, 169, 1e-05, &bn_62_mean, &bn_62_var, &bn_62_weight, &bn_62_bias);
    ReLUOp<DeviceTensor> leaky_62 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_63 = LibDNNConvOp<DeviceTensor> ("conv_63_forward", 86528, &conv_63_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,128,1});
    BatchNormOp<DeviceTensor> bn_63 = BatchNormOp<DeviceTensor> (512, 169, 1e-05, &bn_63_mean, &bn_63_var, &bn_63_weight, &bn_63_bias);
    ReLUOp<DeviceTensor> leaky_63 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_64 = LibDNNConvOp<DeviceTensor> ("conv_64_forward", 173056, &conv_64_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,256,1});
    BatchNormOp<DeviceTensor> bn_64 = BatchNormOp<DeviceTensor> (1024, 169, 1e-05, &bn_64_mean, &bn_64_var, &bn_64_weight, &bn_64_bias);
    ReLUOp<DeviceTensor> leaky_64 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_66 = LibDNNConvOp<DeviceTensor> ("conv_66_forward", 86528, &conv_66_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,128,1});
    BatchNormOp<DeviceTensor> bn_66 = BatchNormOp<DeviceTensor> (512, 169, 1e-05, &bn_66_mean, &bn_66_var, &bn_66_weight, &bn_66_bias);
    ReLUOp<DeviceTensor> leaky_66 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_67 = LibDNNConvOp<DeviceTensor> ("conv_67_forward", 173056, &conv_67_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,256,1});
    BatchNormOp<DeviceTensor> bn_67 = BatchNormOp<DeviceTensor> (1024, 169, 1e-05, &bn_67_mean, &bn_67_var, &bn_67_weight, &bn_67_bias);
    ReLUOp<DeviceTensor> leaky_67 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_69 = LibDNNConvOp<DeviceTensor> ("conv_69_forward", 86528, &conv_69_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,128,1});
    BatchNormOp<DeviceTensor> bn_69 = BatchNormOp<DeviceTensor> (512, 169, 1e-05, &bn_69_mean, &bn_69_var, &bn_69_weight, &bn_69_bias);
    ReLUOp<DeviceTensor> leaky_69 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_70 = LibDNNConvOp<DeviceTensor> ("conv_70_forward", 173056, &conv_70_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,256,1});
    BatchNormOp<DeviceTensor> bn_70 = BatchNormOp<DeviceTensor> (1024, 169, 1e-05, &bn_70_mean, &bn_70_var, &bn_70_weight, &bn_70_bias);
    ReLUOp<DeviceTensor> leaky_70 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_72 = LibDNNConvOp<DeviceTensor> ("conv_72_forward", 86528, &conv_72_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,128,1});
    BatchNormOp<DeviceTensor> bn_72 = BatchNormOp<DeviceTensor> (512, 169, 1e-05, &bn_72_mean, &bn_72_var, &bn_72_weight, &bn_72_bias);
    ReLUOp<DeviceTensor> leaky_72 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_73 = LibDNNConvOp<DeviceTensor> ("conv_73_forward", 173056, &conv_73_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,256,1});
    BatchNormOp<DeviceTensor> bn_73 = BatchNormOp<DeviceTensor> (1024, 169, 1e-05, &bn_73_mean, &bn_73_var, &bn_73_weight, &bn_73_bias);
    ReLUOp<DeviceTensor> leaky_73 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_75 = LibDNNConvOp<DeviceTensor> ("conv_75_forward", 86528, &conv_75_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,128,1});
    BatchNormOp<DeviceTensor> bn_75 = BatchNormOp<DeviceTensor> (512, 169, 1e-05, &bn_75_mean, &bn_75_var, &bn_75_weight, &bn_75_bias);
    ReLUOp<DeviceTensor> leaky_75 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_76 = LibDNNConvOp<DeviceTensor> ("conv_76_forward", 173056, &conv_76_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,256,1});
    BatchNormOp<DeviceTensor> bn_76 = BatchNormOp<DeviceTensor> (1024, 169, 1e-05, &bn_76_mean, &bn_76_var, &bn_76_weight, &bn_76_bias);
    ReLUOp<DeviceTensor> leaky_76 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_77 = LibDNNConvOp<DeviceTensor> ("conv_77_forward", 86528, &conv_77_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,128,1});
    BatchNormOp<DeviceTensor> bn_77 = BatchNormOp<DeviceTensor> (512, 169, 1e-05, &bn_77_mean, &bn_77_var, &bn_77_weight, &bn_77_bias);
    ReLUOp<DeviceTensor> leaky_77 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_78 = LibDNNConvOp<DeviceTensor> ("conv_78_forward", 173056, &conv_78_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,256,1});
    BatchNormOp<DeviceTensor> bn_78 = BatchNormOp<DeviceTensor> (1024, 169, 1e-05, &bn_78_mean, &bn_78_var, &bn_78_weight, &bn_78_bias);
    ReLUOp<DeviceTensor> leaky_78 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_79 = LibDNNConvOp<DeviceTensor> ("conv_79_forward", 86528, &conv_79_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,128,1});
    BatchNormOp<DeviceTensor> bn_79 = BatchNormOp<DeviceTensor> (512, 169, 1e-05, &bn_79_mean, &bn_79_var, &bn_79_weight, &bn_79_bias);
    ReLUOp<DeviceTensor> leaky_79 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_80 = LibDNNConvOp<DeviceTensor> ("conv_80_forward", 173056, &conv_80_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,256,1});
    BatchNormOp<DeviceTensor> bn_80 = BatchNormOp<DeviceTensor> (1024, 169, 1e-05, &bn_80_mean, &bn_80_var, &bn_80_weight, &bn_80_bias);
    ReLUOp<DeviceTensor> leaky_80 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_81 = LibDNNConvOp<DeviceTensor> ("conv_81_forward", 43095, &conv_81_weight, &conv_81_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,64,1});
    LibDNNConvOp<DeviceTensor> conv_84 = LibDNNConvOp<DeviceTensor> ("conv_84_forward", 43264, &conv_84_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,64,1});
    UpSampling2D<DeviceTensor> upsampling_85 = UpSampling2D<DeviceTensor>(2, 13, 13);
    BatchNormOp<DeviceTensor> bn_84 = BatchNormOp<DeviceTensor> (256, 169, 1e-05, &bn_84_mean, &bn_84_var, &bn_84_weight, &bn_84_bias);
    ReLUOp<DeviceTensor> leaky_84 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_87 = LibDNNConvOp<DeviceTensor> ("conv_87_forward", 173056, &conv_87_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    BatchNormOp<DeviceTensor> bn_87 = BatchNormOp<DeviceTensor> (256, 676, 1e-05, &bn_87_mean, &bn_87_var, &bn_87_weight, &bn_87_bias);
    ReLUOp<DeviceTensor> leaky_87 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_88 = LibDNNConvOp<DeviceTensor> ("conv_88_forward", 346112, &conv_88_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> bn_88 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &bn_88_mean, &bn_88_var, &bn_88_weight, &bn_88_bias);
    ReLUOp<DeviceTensor> leaky_88 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_89 = LibDNNConvOp<DeviceTensor> ("conv_89_forward", 173056, &conv_89_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    BatchNormOp<DeviceTensor> bn_89 = BatchNormOp<DeviceTensor> (256, 676, 1e-05, &bn_89_mean, &bn_89_var, &bn_89_weight, &bn_89_bias);
    ReLUOp<DeviceTensor> leaky_89 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_90 = LibDNNConvOp<DeviceTensor> ("conv_90_forward", 346112, &conv_90_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> bn_90 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &bn_90_mean, &bn_90_var, &bn_90_weight, &bn_90_bias);
    ReLUOp<DeviceTensor> leaky_90 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_91 = LibDNNConvOp<DeviceTensor> ("conv_91_forward", 173056, &conv_91_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    BatchNormOp<DeviceTensor> bn_91 = BatchNormOp<DeviceTensor> (256, 676, 1e-05, &bn_91_mean, &bn_91_var, &bn_91_weight, &bn_91_bias);
    ReLUOp<DeviceTensor> leaky_91 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_92 = LibDNNConvOp<DeviceTensor> ("conv_92_forward", 346112, &conv_92_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> bn_92 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &bn_92_mean, &bn_92_var, &bn_92_weight, &bn_92_bias);
    ReLUOp<DeviceTensor> leaky_92 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_93 = LibDNNConvOp<DeviceTensor> ("conv_93_forward", 172380, &conv_93_weight, &conv_93_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    LibDNNConvOp<DeviceTensor> conv_96 = LibDNNConvOp<DeviceTensor> ("conv_96_forward", 86528, &conv_96_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,32,1});
    BatchNormOp<DeviceTensor> bn_96 = BatchNormOp<DeviceTensor> (128, 676, 1e-05, &bn_96_mean, &bn_96_var, &bn_96_weight, &bn_96_bias);
    ReLUOp<DeviceTensor> leaky_96 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    UpSampling2D<DeviceTensor> upsampling_97 = UpSampling2D<DeviceTensor>(2, 26, 26);
    LibDNNConvOp<DeviceTensor> conv_99 = LibDNNConvOp<DeviceTensor> ("conv_99_forward", 346112, &conv_99_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,32,1});
    BatchNormOp<DeviceTensor> bn_99 = BatchNormOp<DeviceTensor> (128, 2704, 1e-05, &bn_99_mean, &bn_99_var, &bn_99_weight, &bn_99_bias);
    ReLUOp<DeviceTensor> leaky_99 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_100 = LibDNNConvOp<DeviceTensor> ("conv_100_forward", 692224, &conv_100_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> bn_100 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &bn_100_mean, &bn_100_var, &bn_100_weight, &bn_100_bias);
    ReLUOp<DeviceTensor> leaky_100 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_101 = LibDNNConvOp<DeviceTensor> ("conv_101_forward", 346112, &conv_101_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,32,1});
    BatchNormOp<DeviceTensor> bn_101 = BatchNormOp<DeviceTensor> (128, 2704, 1e-05, &bn_101_mean, &bn_101_var, &bn_101_weight, &bn_101_bias);
    ReLUOp<DeviceTensor> leaky_101 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_102 = LibDNNConvOp<DeviceTensor> ("conv_102_forward", 692224, &conv_102_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> bn_102 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &bn_102_mean, &bn_102_var, &bn_102_weight, &bn_102_bias);
    ReLUOp<DeviceTensor> leaky_102 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_103 = LibDNNConvOp<DeviceTensor> ("conv_103_forward", 346112, &conv_103_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,32,1});
    BatchNormOp<DeviceTensor> bn_103 = BatchNormOp<DeviceTensor> (128, 2704, 1e-05, &bn_103_mean, &bn_103_var, &bn_103_weight, &bn_103_bias);
    ReLUOp<DeviceTensor> leaky_103 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_104 = LibDNNConvOp<DeviceTensor> ("conv_104_forward", 692224, &conv_104_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> bn_104 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &bn_104_mean, &bn_104_var, &bn_104_weight, &bn_104_bias);
    ReLUOp<DeviceTensor> leaky_104 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_105 = LibDNNConvOp<DeviceTensor> ("conv_105_forward", 689520, &conv_105_weight, &conv_105_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});

};


} //namespace hypertea