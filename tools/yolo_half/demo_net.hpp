#include "hypertea/hypertea.hpp"
#include "kernels/conv_kernel.cl"

namespace hypertea {

using DeviceTensor = TensorGPU<half>;

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
            if (half2float_impl(cpu_data.get()[anchor_offset + i]) > confidence_inv_sigmoid) {
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
            output_data[i * out_num + n] = half2float_impl(cpu_data.get()[(anchor_index[n] * bbox_attrs + i) * grid_square + pos_index[n]]);
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

        // compile_opencl_kernels(conv_opencl_funcs, " ", true, "/sdcard/hypertea_ws/yolo_half/");
        
        load_opencl_kernels("/sdcard/hypertea_ws/yolo_half/prebuilt_math_program", "/sdcard/hypertea_ws/yolo_half/prebuilt_conv_program", " ");

        auto all_weights = load_weights<half>(param_file, 62001757);

        std::cout << "we are go here !!" << std::endl;

        conv_0_weight.copy_from_ptr(all_weights + 0);
        bn_0_mean.copy_from_ptr(all_weights + 864);
        bn_0_var.copy_from_ptr(all_weights + 896);
        bn_0_weight.copy_from_ptr(all_weights + 928);
        bn_0_bias.copy_from_ptr(all_weights + 960);
        conv_1_weight.copy_from_ptr(all_weights + 992);
        bn_1_mean.copy_from_ptr(all_weights + 19424);
        bn_1_var.copy_from_ptr(all_weights + 19488);
        bn_1_weight.copy_from_ptr(all_weights + 19552);
        bn_1_bias.copy_from_ptr(all_weights + 19616);
        conv_2_weight.copy_from_ptr(all_weights + 19680);
        bn_2_mean.copy_from_ptr(all_weights + 21728);
        bn_2_var.copy_from_ptr(all_weights + 21760);
        bn_2_weight.copy_from_ptr(all_weights + 21792);
        bn_2_bias.copy_from_ptr(all_weights + 21824);
        conv_3_weight.copy_from_ptr(all_weights + 21856);
        bn_3_mean.copy_from_ptr(all_weights + 40288);
        bn_3_var.copy_from_ptr(all_weights + 40352);
        bn_3_weight.copy_from_ptr(all_weights + 40416);
        bn_3_bias.copy_from_ptr(all_weights + 40480);
        conv_5_weight.copy_from_ptr(all_weights + 40544);
        bn_5_mean.copy_from_ptr(all_weights + 114272);
        bn_5_var.copy_from_ptr(all_weights + 114400);
        bn_5_weight.copy_from_ptr(all_weights + 114528);
        bn_5_bias.copy_from_ptr(all_weights + 114656);
        conv_6_weight.copy_from_ptr(all_weights + 114784);
        bn_6_mean.copy_from_ptr(all_weights + 122976);
        bn_6_var.copy_from_ptr(all_weights + 123040);
        bn_6_weight.copy_from_ptr(all_weights + 123104);
        bn_6_bias.copy_from_ptr(all_weights + 123168);
        conv_7_weight.copy_from_ptr(all_weights + 123232);
        bn_7_mean.copy_from_ptr(all_weights + 196960);
        bn_7_var.copy_from_ptr(all_weights + 197088);
        bn_7_weight.copy_from_ptr(all_weights + 197216);
        bn_7_bias.copy_from_ptr(all_weights + 197344);
        conv_9_weight.copy_from_ptr(all_weights + 197472);
        bn_9_mean.copy_from_ptr(all_weights + 205664);
        bn_9_var.copy_from_ptr(all_weights + 205728);
        bn_9_weight.copy_from_ptr(all_weights + 205792);
        bn_9_bias.copy_from_ptr(all_weights + 205856);
        conv_10_weight.copy_from_ptr(all_weights + 205920);
        bn_10_mean.copy_from_ptr(all_weights + 279648);
        bn_10_var.copy_from_ptr(all_weights + 279776);
        bn_10_weight.copy_from_ptr(all_weights + 279904);
        bn_10_bias.copy_from_ptr(all_weights + 280032);
        conv_12_weight.copy_from_ptr(all_weights + 280160);
        bn_12_mean.copy_from_ptr(all_weights + 575072);
        bn_12_var.copy_from_ptr(all_weights + 575328);
        bn_12_weight.copy_from_ptr(all_weights + 575584);
        bn_12_bias.copy_from_ptr(all_weights + 575840);
        conv_13_weight.copy_from_ptr(all_weights + 576096);
        bn_13_mean.copy_from_ptr(all_weights + 608864);
        bn_13_var.copy_from_ptr(all_weights + 608992);
        bn_13_weight.copy_from_ptr(all_weights + 609120);
        bn_13_bias.copy_from_ptr(all_weights + 609248);
        conv_14_weight.copy_from_ptr(all_weights + 609376);
        bn_14_mean.copy_from_ptr(all_weights + 904288);
        bn_14_var.copy_from_ptr(all_weights + 904544);
        bn_14_weight.copy_from_ptr(all_weights + 904800);
        bn_14_bias.copy_from_ptr(all_weights + 905056);
        conv_16_weight.copy_from_ptr(all_weights + 905312);
        bn_16_mean.copy_from_ptr(all_weights + 938080);
        bn_16_var.copy_from_ptr(all_weights + 938208);
        bn_16_weight.copy_from_ptr(all_weights + 938336);
        bn_16_bias.copy_from_ptr(all_weights + 938464);
        conv_17_weight.copy_from_ptr(all_weights + 938592);
        bn_17_mean.copy_from_ptr(all_weights + 1233504);
        bn_17_var.copy_from_ptr(all_weights + 1233760);
        bn_17_weight.copy_from_ptr(all_weights + 1234016);
        bn_17_bias.copy_from_ptr(all_weights + 1234272);
        conv_19_weight.copy_from_ptr(all_weights + 1234528);
        bn_19_mean.copy_from_ptr(all_weights + 1267296);
        bn_19_var.copy_from_ptr(all_weights + 1267424);
        bn_19_weight.copy_from_ptr(all_weights + 1267552);
        bn_19_bias.copy_from_ptr(all_weights + 1267680);
        conv_20_weight.copy_from_ptr(all_weights + 1267808);
        bn_20_mean.copy_from_ptr(all_weights + 1562720);
        bn_20_var.copy_from_ptr(all_weights + 1562976);
        bn_20_weight.copy_from_ptr(all_weights + 1563232);
        bn_20_bias.copy_from_ptr(all_weights + 1563488);
        conv_22_weight.copy_from_ptr(all_weights + 1563744);
        bn_22_mean.copy_from_ptr(all_weights + 1596512);
        bn_22_var.copy_from_ptr(all_weights + 1596640);
        bn_22_weight.copy_from_ptr(all_weights + 1596768);
        bn_22_bias.copy_from_ptr(all_weights + 1596896);
        conv_23_weight.copy_from_ptr(all_weights + 1597024);
        bn_23_mean.copy_from_ptr(all_weights + 1891936);
        bn_23_var.copy_from_ptr(all_weights + 1892192);
        bn_23_weight.copy_from_ptr(all_weights + 1892448);
        bn_23_bias.copy_from_ptr(all_weights + 1892704);
        conv_25_weight.copy_from_ptr(all_weights + 1892960);
        bn_25_mean.copy_from_ptr(all_weights + 1925728);
        bn_25_var.copy_from_ptr(all_weights + 1925856);
        bn_25_weight.copy_from_ptr(all_weights + 1925984);
        bn_25_bias.copy_from_ptr(all_weights + 1926112);
        conv_26_weight.copy_from_ptr(all_weights + 1926240);
        bn_26_mean.copy_from_ptr(all_weights + 2221152);
        bn_26_var.copy_from_ptr(all_weights + 2221408);
        bn_26_weight.copy_from_ptr(all_weights + 2221664);
        bn_26_bias.copy_from_ptr(all_weights + 2221920);
        conv_28_weight.copy_from_ptr(all_weights + 2222176);
        bn_28_mean.copy_from_ptr(all_weights + 2254944);
        bn_28_var.copy_from_ptr(all_weights + 2255072);
        bn_28_weight.copy_from_ptr(all_weights + 2255200);
        bn_28_bias.copy_from_ptr(all_weights + 2255328);
        conv_29_weight.copy_from_ptr(all_weights + 2255456);
        bn_29_mean.copy_from_ptr(all_weights + 2550368);
        bn_29_var.copy_from_ptr(all_weights + 2550624);
        bn_29_weight.copy_from_ptr(all_weights + 2550880);
        bn_29_bias.copy_from_ptr(all_weights + 2551136);
        conv_31_weight.copy_from_ptr(all_weights + 2551392);
        bn_31_mean.copy_from_ptr(all_weights + 2584160);
        bn_31_var.copy_from_ptr(all_weights + 2584288);
        bn_31_weight.copy_from_ptr(all_weights + 2584416);
        bn_31_bias.copy_from_ptr(all_weights + 2584544);
        conv_32_weight.copy_from_ptr(all_weights + 2584672);
        bn_32_mean.copy_from_ptr(all_weights + 2879584);
        bn_32_var.copy_from_ptr(all_weights + 2879840);
        bn_32_weight.copy_from_ptr(all_weights + 2880096);
        bn_32_bias.copy_from_ptr(all_weights + 2880352);
        conv_34_weight.copy_from_ptr(all_weights + 2880608);
        bn_34_mean.copy_from_ptr(all_weights + 2913376);
        bn_34_var.copy_from_ptr(all_weights + 2913504);
        bn_34_weight.copy_from_ptr(all_weights + 2913632);
        bn_34_bias.copy_from_ptr(all_weights + 2913760);
        conv_35_weight.copy_from_ptr(all_weights + 2913888);
        bn_35_mean.copy_from_ptr(all_weights + 3208800);
        bn_35_var.copy_from_ptr(all_weights + 3209056);
        bn_35_weight.copy_from_ptr(all_weights + 3209312);
        bn_35_bias.copy_from_ptr(all_weights + 3209568);
        conv_37_weight.copy_from_ptr(all_weights + 3209824);
        bn_37_mean.copy_from_ptr(all_weights + 4389472);
        bn_37_var.copy_from_ptr(all_weights + 4389984);
        bn_37_weight.copy_from_ptr(all_weights + 4390496);
        bn_37_bias.copy_from_ptr(all_weights + 4391008);
        conv_38_weight.copy_from_ptr(all_weights + 4391520);
        bn_38_mean.copy_from_ptr(all_weights + 4522592);
        bn_38_var.copy_from_ptr(all_weights + 4522848);
        bn_38_weight.copy_from_ptr(all_weights + 4523104);
        bn_38_bias.copy_from_ptr(all_weights + 4523360);
        conv_39_weight.copy_from_ptr(all_weights + 4523616);
        bn_39_mean.copy_from_ptr(all_weights + 5703264);
        bn_39_var.copy_from_ptr(all_weights + 5703776);
        bn_39_weight.copy_from_ptr(all_weights + 5704288);
        bn_39_bias.copy_from_ptr(all_weights + 5704800);
        conv_41_weight.copy_from_ptr(all_weights + 5705312);
        bn_41_mean.copy_from_ptr(all_weights + 5836384);
        bn_41_var.copy_from_ptr(all_weights + 5836640);
        bn_41_weight.copy_from_ptr(all_weights + 5836896);
        bn_41_bias.copy_from_ptr(all_weights + 5837152);
        conv_42_weight.copy_from_ptr(all_weights + 5837408);
        bn_42_mean.copy_from_ptr(all_weights + 7017056);
        bn_42_var.copy_from_ptr(all_weights + 7017568);
        bn_42_weight.copy_from_ptr(all_weights + 7018080);
        bn_42_bias.copy_from_ptr(all_weights + 7018592);
        conv_44_weight.copy_from_ptr(all_weights + 7019104);
        bn_44_mean.copy_from_ptr(all_weights + 7150176);
        bn_44_var.copy_from_ptr(all_weights + 7150432);
        bn_44_weight.copy_from_ptr(all_weights + 7150688);
        bn_44_bias.copy_from_ptr(all_weights + 7150944);
        conv_45_weight.copy_from_ptr(all_weights + 7151200);
        bn_45_mean.copy_from_ptr(all_weights + 8330848);
        bn_45_var.copy_from_ptr(all_weights + 8331360);
        bn_45_weight.copy_from_ptr(all_weights + 8331872);
        bn_45_bias.copy_from_ptr(all_weights + 8332384);
        conv_47_weight.copy_from_ptr(all_weights + 8332896);
        bn_47_mean.copy_from_ptr(all_weights + 8463968);
        bn_47_var.copy_from_ptr(all_weights + 8464224);
        bn_47_weight.copy_from_ptr(all_weights + 8464480);
        bn_47_bias.copy_from_ptr(all_weights + 8464736);
        conv_48_weight.copy_from_ptr(all_weights + 8464992);
        bn_48_mean.copy_from_ptr(all_weights + 9644640);
        bn_48_var.copy_from_ptr(all_weights + 9645152);
        bn_48_weight.copy_from_ptr(all_weights + 9645664);
        bn_48_bias.copy_from_ptr(all_weights + 9646176);
        conv_50_weight.copy_from_ptr(all_weights + 9646688);
        bn_50_mean.copy_from_ptr(all_weights + 9777760);
        bn_50_var.copy_from_ptr(all_weights + 9778016);
        bn_50_weight.copy_from_ptr(all_weights + 9778272);
        bn_50_bias.copy_from_ptr(all_weights + 9778528);
        conv_51_weight.copy_from_ptr(all_weights + 9778784);
        bn_51_mean.copy_from_ptr(all_weights + 10958432);
        bn_51_var.copy_from_ptr(all_weights + 10958944);
        bn_51_weight.copy_from_ptr(all_weights + 10959456);
        bn_51_bias.copy_from_ptr(all_weights + 10959968);
        conv_53_weight.copy_from_ptr(all_weights + 10960480);
        bn_53_mean.copy_from_ptr(all_weights + 11091552);
        bn_53_var.copy_from_ptr(all_weights + 11091808);
        bn_53_weight.copy_from_ptr(all_weights + 11092064);
        bn_53_bias.copy_from_ptr(all_weights + 11092320);
        conv_54_weight.copy_from_ptr(all_weights + 11092576);
        bn_54_mean.copy_from_ptr(all_weights + 12272224);
        bn_54_var.copy_from_ptr(all_weights + 12272736);
        bn_54_weight.copy_from_ptr(all_weights + 12273248);
        bn_54_bias.copy_from_ptr(all_weights + 12273760);
        conv_56_weight.copy_from_ptr(all_weights + 12274272);
        bn_56_mean.copy_from_ptr(all_weights + 12405344);
        bn_56_var.copy_from_ptr(all_weights + 12405600);
        bn_56_weight.copy_from_ptr(all_weights + 12405856);
        bn_56_bias.copy_from_ptr(all_weights + 12406112);
        conv_57_weight.copy_from_ptr(all_weights + 12406368);
        bn_57_mean.copy_from_ptr(all_weights + 13586016);
        bn_57_var.copy_from_ptr(all_weights + 13586528);
        bn_57_weight.copy_from_ptr(all_weights + 13587040);
        bn_57_bias.copy_from_ptr(all_weights + 13587552);
        conv_59_weight.copy_from_ptr(all_weights + 13588064);
        bn_59_mean.copy_from_ptr(all_weights + 13719136);
        bn_59_var.copy_from_ptr(all_weights + 13719392);
        bn_59_weight.copy_from_ptr(all_weights + 13719648);
        bn_59_bias.copy_from_ptr(all_weights + 13719904);
        conv_60_weight.copy_from_ptr(all_weights + 13720160);
        bn_60_mean.copy_from_ptr(all_weights + 14899808);
        bn_60_var.copy_from_ptr(all_weights + 14900320);
        bn_60_weight.copy_from_ptr(all_weights + 14900832);
        bn_60_bias.copy_from_ptr(all_weights + 14901344);
        conv_62_weight.copy_from_ptr(all_weights + 14901856);
        bn_62_mean.copy_from_ptr(all_weights + 19620448);
        bn_62_var.copy_from_ptr(all_weights + 19621472);
        bn_62_weight.copy_from_ptr(all_weights + 19622496);
        bn_62_bias.copy_from_ptr(all_weights + 19623520);
        conv_63_weight.copy_from_ptr(all_weights + 19624544);
        bn_63_mean.copy_from_ptr(all_weights + 20148832);
        bn_63_var.copy_from_ptr(all_weights + 20149344);
        bn_63_weight.copy_from_ptr(all_weights + 20149856);
        bn_63_bias.copy_from_ptr(all_weights + 20150368);
        conv_64_weight.copy_from_ptr(all_weights + 20150880);
        bn_64_mean.copy_from_ptr(all_weights + 24869472);
        bn_64_var.copy_from_ptr(all_weights + 24870496);
        bn_64_weight.copy_from_ptr(all_weights + 24871520);
        bn_64_bias.copy_from_ptr(all_weights + 24872544);
        conv_66_weight.copy_from_ptr(all_weights + 24873568);
        bn_66_mean.copy_from_ptr(all_weights + 25397856);
        bn_66_var.copy_from_ptr(all_weights + 25398368);
        bn_66_weight.copy_from_ptr(all_weights + 25398880);
        bn_66_bias.copy_from_ptr(all_weights + 25399392);
        conv_67_weight.copy_from_ptr(all_weights + 25399904);
        bn_67_mean.copy_from_ptr(all_weights + 30118496);
        bn_67_var.copy_from_ptr(all_weights + 30119520);
        bn_67_weight.copy_from_ptr(all_weights + 30120544);
        bn_67_bias.copy_from_ptr(all_weights + 30121568);
        conv_69_weight.copy_from_ptr(all_weights + 30122592);
        bn_69_mean.copy_from_ptr(all_weights + 30646880);
        bn_69_var.copy_from_ptr(all_weights + 30647392);
        bn_69_weight.copy_from_ptr(all_weights + 30647904);
        bn_69_bias.copy_from_ptr(all_weights + 30648416);
        conv_70_weight.copy_from_ptr(all_weights + 30648928);
        bn_70_mean.copy_from_ptr(all_weights + 35367520);
        bn_70_var.copy_from_ptr(all_weights + 35368544);
        bn_70_weight.copy_from_ptr(all_weights + 35369568);
        bn_70_bias.copy_from_ptr(all_weights + 35370592);
        conv_72_weight.copy_from_ptr(all_weights + 35371616);
        bn_72_mean.copy_from_ptr(all_weights + 35895904);
        bn_72_var.copy_from_ptr(all_weights + 35896416);
        bn_72_weight.copy_from_ptr(all_weights + 35896928);
        bn_72_bias.copy_from_ptr(all_weights + 35897440);
        conv_73_weight.copy_from_ptr(all_weights + 35897952);
        bn_73_mean.copy_from_ptr(all_weights + 40616544);
        bn_73_var.copy_from_ptr(all_weights + 40617568);
        bn_73_weight.copy_from_ptr(all_weights + 40618592);
        bn_73_bias.copy_from_ptr(all_weights + 40619616);
        conv_75_weight.copy_from_ptr(all_weights + 40620640);
        bn_75_mean.copy_from_ptr(all_weights + 41144928);
        bn_75_var.copy_from_ptr(all_weights + 41145440);
        bn_75_weight.copy_from_ptr(all_weights + 41145952);
        bn_75_bias.copy_from_ptr(all_weights + 41146464);
        conv_76_weight.copy_from_ptr(all_weights + 41146976);
        bn_76_mean.copy_from_ptr(all_weights + 45865568);
        bn_76_var.copy_from_ptr(all_weights + 45866592);
        bn_76_weight.copy_from_ptr(all_weights + 45867616);
        bn_76_bias.copy_from_ptr(all_weights + 45868640);
        conv_77_weight.copy_from_ptr(all_weights + 45869664);
        bn_77_mean.copy_from_ptr(all_weights + 46393952);
        bn_77_var.copy_from_ptr(all_weights + 46394464);
        bn_77_weight.copy_from_ptr(all_weights + 46394976);
        bn_77_bias.copy_from_ptr(all_weights + 46395488);
        conv_78_weight.copy_from_ptr(all_weights + 46396000);
        bn_78_mean.copy_from_ptr(all_weights + 51114592);
        bn_78_var.copy_from_ptr(all_weights + 51115616);
        bn_78_weight.copy_from_ptr(all_weights + 51116640);
        bn_78_bias.copy_from_ptr(all_weights + 51117664);
        conv_79_weight.copy_from_ptr(all_weights + 51118688);
        bn_79_mean.copy_from_ptr(all_weights + 51642976);
        bn_79_var.copy_from_ptr(all_weights + 51643488);
        bn_79_weight.copy_from_ptr(all_weights + 51644000);
        bn_79_bias.copy_from_ptr(all_weights + 51644512);
        conv_80_weight.copy_from_ptr(all_weights + 51645024);
        bn_80_mean.copy_from_ptr(all_weights + 56363616);
        bn_80_var.copy_from_ptr(all_weights + 56364640);
        bn_80_weight.copy_from_ptr(all_weights + 56365664);
        bn_80_bias.copy_from_ptr(all_weights + 56366688);
        conv_81_bias.copy_from_ptr(all_weights + 56367712);
        conv_81_weight.copy_from_ptr(all_weights + 56367967);
        conv_84_weight.copy_from_ptr(all_weights + 56629087);
        bn_84_mean.copy_from_ptr(all_weights + 56760159);
        bn_84_var.copy_from_ptr(all_weights + 56760415);
        bn_84_weight.copy_from_ptr(all_weights + 56760671);
        bn_84_bias.copy_from_ptr(all_weights + 56760927);
        conv_87_weight.copy_from_ptr(all_weights + 56761183);
        bn_87_mean.copy_from_ptr(all_weights + 56957791);
        bn_87_var.copy_from_ptr(all_weights + 56958047);
        bn_87_weight.copy_from_ptr(all_weights + 56958303);
        bn_87_bias.copy_from_ptr(all_weights + 56958559);
        conv_88_weight.copy_from_ptr(all_weights + 56958815);
        bn_88_mean.copy_from_ptr(all_weights + 58138463);
        bn_88_var.copy_from_ptr(all_weights + 58138975);
        bn_88_weight.copy_from_ptr(all_weights + 58139487);
        bn_88_bias.copy_from_ptr(all_weights + 58139999);
        conv_89_weight.copy_from_ptr(all_weights + 58140511);
        bn_89_mean.copy_from_ptr(all_weights + 58271583);
        bn_89_var.copy_from_ptr(all_weights + 58271839);
        bn_89_weight.copy_from_ptr(all_weights + 58272095);
        bn_89_bias.copy_from_ptr(all_weights + 58272351);
        conv_90_weight.copy_from_ptr(all_weights + 58272607);
        bn_90_mean.copy_from_ptr(all_weights + 59452255);
        bn_90_var.copy_from_ptr(all_weights + 59452767);
        bn_90_weight.copy_from_ptr(all_weights + 59453279);
        bn_90_bias.copy_from_ptr(all_weights + 59453791);
        conv_91_weight.copy_from_ptr(all_weights + 59454303);
        bn_91_mean.copy_from_ptr(all_weights + 59585375);
        bn_91_var.copy_from_ptr(all_weights + 59585631);
        bn_91_weight.copy_from_ptr(all_weights + 59585887);
        bn_91_bias.copy_from_ptr(all_weights + 59586143);
        conv_92_weight.copy_from_ptr(all_weights + 59586399);
        bn_92_mean.copy_from_ptr(all_weights + 60766047);
        bn_92_var.copy_from_ptr(all_weights + 60766559);
        bn_92_weight.copy_from_ptr(all_weights + 60767071);
        bn_92_bias.copy_from_ptr(all_weights + 60767583);
        conv_93_bias.copy_from_ptr(all_weights + 60768095);
        conv_93_weight.copy_from_ptr(all_weights + 60768350);
        conv_96_weight.copy_from_ptr(all_weights + 60898910);
        bn_96_mean.copy_from_ptr(all_weights + 60931678);
        bn_96_var.copy_from_ptr(all_weights + 60931806);
        bn_96_weight.copy_from_ptr(all_weights + 60931934);
        bn_96_bias.copy_from_ptr(all_weights + 60932062);
        conv_99_weight.copy_from_ptr(all_weights + 60932190);
        bn_99_mean.copy_from_ptr(all_weights + 60981342);
        bn_99_var.copy_from_ptr(all_weights + 60981470);
        bn_99_weight.copy_from_ptr(all_weights + 60981598);
        bn_99_bias.copy_from_ptr(all_weights + 60981726);
        conv_100_weight.copy_from_ptr(all_weights + 60981854);
        bn_100_mean.copy_from_ptr(all_weights + 61276766);
        bn_100_var.copy_from_ptr(all_weights + 61277022);
        bn_100_weight.copy_from_ptr(all_weights + 61277278);
        bn_100_bias.copy_from_ptr(all_weights + 61277534);
        conv_101_weight.copy_from_ptr(all_weights + 61277790);
        bn_101_mean.copy_from_ptr(all_weights + 61310558);
        bn_101_var.copy_from_ptr(all_weights + 61310686);
        bn_101_weight.copy_from_ptr(all_weights + 61310814);
        bn_101_bias.copy_from_ptr(all_weights + 61310942);
        conv_102_weight.copy_from_ptr(all_weights + 61311070);
        bn_102_mean.copy_from_ptr(all_weights + 61605982);
        bn_102_var.copy_from_ptr(all_weights + 61606238);
        bn_102_weight.copy_from_ptr(all_weights + 61606494);
        bn_102_bias.copy_from_ptr(all_weights + 61606750);
        conv_103_weight.copy_from_ptr(all_weights + 61607006);
        bn_103_mean.copy_from_ptr(all_weights + 61639774);
        bn_103_var.copy_from_ptr(all_weights + 61639902);
        bn_103_weight.copy_from_ptr(all_weights + 61640030);
        bn_103_bias.copy_from_ptr(all_weights + 61640158);
        conv_104_weight.copy_from_ptr(all_weights + 61640286);
        bn_104_mean.copy_from_ptr(all_weights + 61935198);
        bn_104_var.copy_from_ptr(all_weights + 61935454);
        bn_104_weight.copy_from_ptr(all_weights + 61935710);
        bn_104_bias.copy_from_ptr(all_weights + 61935966);
        conv_105_bias.copy_from_ptr(all_weights + 61936222);
        conv_105_weight.copy_from_ptr(all_weights + 61936477);

        std::cout << "we are go here 1 !!" << std::endl;

        free(all_weights);
    }

    void inference( const std::vector<half> &data_from_user, std::vector<half> &data_to_user) {
        
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
    

    DeviceTensor conv_0_weight = DeviceTensor(864);
    DeviceTensor bn_0_mean = DeviceTensor(32);
    DeviceTensor bn_0_var = DeviceTensor(32);
    DeviceTensor bn_0_weight = DeviceTensor(32);
    DeviceTensor bn_0_bias = DeviceTensor(32);
    DeviceTensor conv_1_weight = DeviceTensor(18432);
    DeviceTensor bn_1_mean = DeviceTensor(64);
    DeviceTensor bn_1_var = DeviceTensor(64);
    DeviceTensor bn_1_weight = DeviceTensor(64);
    DeviceTensor bn_1_bias = DeviceTensor(64);
    DeviceTensor conv_2_weight = DeviceTensor(2048);
    DeviceTensor bn_2_mean = DeviceTensor(32);
    DeviceTensor bn_2_var = DeviceTensor(32);
    DeviceTensor bn_2_weight = DeviceTensor(32);
    DeviceTensor bn_2_bias = DeviceTensor(32);
    DeviceTensor conv_3_weight = DeviceTensor(18432);
    DeviceTensor bn_3_mean = DeviceTensor(64);
    DeviceTensor bn_3_var = DeviceTensor(64);
    DeviceTensor bn_3_weight = DeviceTensor(64);
    DeviceTensor bn_3_bias = DeviceTensor(64);
    DeviceTensor conv_5_weight = DeviceTensor(73728);
    DeviceTensor bn_5_mean = DeviceTensor(128);
    DeviceTensor bn_5_var = DeviceTensor(128);
    DeviceTensor bn_5_weight = DeviceTensor(128);
    DeviceTensor bn_5_bias = DeviceTensor(128);
    DeviceTensor conv_6_weight = DeviceTensor(8192);
    DeviceTensor bn_6_mean = DeviceTensor(64);
    DeviceTensor bn_6_var = DeviceTensor(64);
    DeviceTensor bn_6_weight = DeviceTensor(64);
    DeviceTensor bn_6_bias = DeviceTensor(64);
    DeviceTensor conv_7_weight = DeviceTensor(73728);
    DeviceTensor bn_7_mean = DeviceTensor(128);
    DeviceTensor bn_7_var = DeviceTensor(128);
    DeviceTensor bn_7_weight = DeviceTensor(128);
    DeviceTensor bn_7_bias = DeviceTensor(128);
    DeviceTensor conv_9_weight = DeviceTensor(8192);
    DeviceTensor bn_9_mean = DeviceTensor(64);
    DeviceTensor bn_9_var = DeviceTensor(64);
    DeviceTensor bn_9_weight = DeviceTensor(64);
    DeviceTensor bn_9_bias = DeviceTensor(64);
    DeviceTensor conv_10_weight = DeviceTensor(73728);
    DeviceTensor bn_10_mean = DeviceTensor(128);
    DeviceTensor bn_10_var = DeviceTensor(128);
    DeviceTensor bn_10_weight = DeviceTensor(128);
    DeviceTensor bn_10_bias = DeviceTensor(128);
    DeviceTensor conv_12_weight = DeviceTensor(294912);
    DeviceTensor bn_12_mean = DeviceTensor(256);
    DeviceTensor bn_12_var = DeviceTensor(256);
    DeviceTensor bn_12_weight = DeviceTensor(256);
    DeviceTensor bn_12_bias = DeviceTensor(256);
    DeviceTensor conv_13_weight = DeviceTensor(32768);
    DeviceTensor bn_13_mean = DeviceTensor(128);
    DeviceTensor bn_13_var = DeviceTensor(128);
    DeviceTensor bn_13_weight = DeviceTensor(128);
    DeviceTensor bn_13_bias = DeviceTensor(128);
    DeviceTensor conv_14_weight = DeviceTensor(294912);
    DeviceTensor bn_14_mean = DeviceTensor(256);
    DeviceTensor bn_14_var = DeviceTensor(256);
    DeviceTensor bn_14_weight = DeviceTensor(256);
    DeviceTensor bn_14_bias = DeviceTensor(256);
    DeviceTensor conv_16_weight = DeviceTensor(32768);
    DeviceTensor bn_16_mean = DeviceTensor(128);
    DeviceTensor bn_16_var = DeviceTensor(128);
    DeviceTensor bn_16_weight = DeviceTensor(128);
    DeviceTensor bn_16_bias = DeviceTensor(128);
    DeviceTensor conv_17_weight = DeviceTensor(294912);
    DeviceTensor bn_17_mean = DeviceTensor(256);
    DeviceTensor bn_17_var = DeviceTensor(256);
    DeviceTensor bn_17_weight = DeviceTensor(256);
    DeviceTensor bn_17_bias = DeviceTensor(256);
    DeviceTensor conv_19_weight = DeviceTensor(32768);
    DeviceTensor bn_19_mean = DeviceTensor(128);
    DeviceTensor bn_19_var = DeviceTensor(128);
    DeviceTensor bn_19_weight = DeviceTensor(128);
    DeviceTensor bn_19_bias = DeviceTensor(128);
    DeviceTensor conv_20_weight = DeviceTensor(294912);
    DeviceTensor bn_20_mean = DeviceTensor(256);
    DeviceTensor bn_20_var = DeviceTensor(256);
    DeviceTensor bn_20_weight = DeviceTensor(256);
    DeviceTensor bn_20_bias = DeviceTensor(256);
    DeviceTensor conv_22_weight = DeviceTensor(32768);
    DeviceTensor bn_22_mean = DeviceTensor(128);
    DeviceTensor bn_22_var = DeviceTensor(128);
    DeviceTensor bn_22_weight = DeviceTensor(128);
    DeviceTensor bn_22_bias = DeviceTensor(128);
    DeviceTensor conv_23_weight = DeviceTensor(294912);
    DeviceTensor bn_23_mean = DeviceTensor(256);
    DeviceTensor bn_23_var = DeviceTensor(256);
    DeviceTensor bn_23_weight = DeviceTensor(256);
    DeviceTensor bn_23_bias = DeviceTensor(256);
    DeviceTensor conv_25_weight = DeviceTensor(32768);
    DeviceTensor bn_25_mean = DeviceTensor(128);
    DeviceTensor bn_25_var = DeviceTensor(128);
    DeviceTensor bn_25_weight = DeviceTensor(128);
    DeviceTensor bn_25_bias = DeviceTensor(128);
    DeviceTensor conv_26_weight = DeviceTensor(294912);
    DeviceTensor bn_26_mean = DeviceTensor(256);
    DeviceTensor bn_26_var = DeviceTensor(256);
    DeviceTensor bn_26_weight = DeviceTensor(256);
    DeviceTensor bn_26_bias = DeviceTensor(256);
    DeviceTensor conv_28_weight = DeviceTensor(32768);
    DeviceTensor bn_28_mean = DeviceTensor(128);
    DeviceTensor bn_28_var = DeviceTensor(128);
    DeviceTensor bn_28_weight = DeviceTensor(128);
    DeviceTensor bn_28_bias = DeviceTensor(128);
    DeviceTensor conv_29_weight = DeviceTensor(294912);
    DeviceTensor bn_29_mean = DeviceTensor(256);
    DeviceTensor bn_29_var = DeviceTensor(256);
    DeviceTensor bn_29_weight = DeviceTensor(256);
    DeviceTensor bn_29_bias = DeviceTensor(256);
    DeviceTensor conv_31_weight = DeviceTensor(32768);
    DeviceTensor bn_31_mean = DeviceTensor(128);
    DeviceTensor bn_31_var = DeviceTensor(128);
    DeviceTensor bn_31_weight = DeviceTensor(128);
    DeviceTensor bn_31_bias = DeviceTensor(128);
    DeviceTensor conv_32_weight = DeviceTensor(294912);
    DeviceTensor bn_32_mean = DeviceTensor(256);
    DeviceTensor bn_32_var = DeviceTensor(256);
    DeviceTensor bn_32_weight = DeviceTensor(256);
    DeviceTensor bn_32_bias = DeviceTensor(256);
    DeviceTensor conv_34_weight = DeviceTensor(32768);
    DeviceTensor bn_34_mean = DeviceTensor(128);
    DeviceTensor bn_34_var = DeviceTensor(128);
    DeviceTensor bn_34_weight = DeviceTensor(128);
    DeviceTensor bn_34_bias = DeviceTensor(128);
    DeviceTensor conv_35_weight = DeviceTensor(294912);
    DeviceTensor bn_35_mean = DeviceTensor(256);
    DeviceTensor bn_35_var = DeviceTensor(256);
    DeviceTensor bn_35_weight = DeviceTensor(256);
    DeviceTensor bn_35_bias = DeviceTensor(256);
    DeviceTensor conv_37_weight = DeviceTensor(1179648);
    DeviceTensor bn_37_mean = DeviceTensor(512);
    DeviceTensor bn_37_var = DeviceTensor(512);
    DeviceTensor bn_37_weight = DeviceTensor(512);
    DeviceTensor bn_37_bias = DeviceTensor(512);
    DeviceTensor conv_38_weight = DeviceTensor(131072);
    DeviceTensor bn_38_mean = DeviceTensor(256);
    DeviceTensor bn_38_var = DeviceTensor(256);
    DeviceTensor bn_38_weight = DeviceTensor(256);
    DeviceTensor bn_38_bias = DeviceTensor(256);
    DeviceTensor conv_39_weight = DeviceTensor(1179648);
    DeviceTensor bn_39_mean = DeviceTensor(512);
    DeviceTensor bn_39_var = DeviceTensor(512);
    DeviceTensor bn_39_weight = DeviceTensor(512);
    DeviceTensor bn_39_bias = DeviceTensor(512);
    DeviceTensor conv_41_weight = DeviceTensor(131072);
    DeviceTensor bn_41_mean = DeviceTensor(256);
    DeviceTensor bn_41_var = DeviceTensor(256);
    DeviceTensor bn_41_weight = DeviceTensor(256);
    DeviceTensor bn_41_bias = DeviceTensor(256);
    DeviceTensor conv_42_weight = DeviceTensor(1179648);
    DeviceTensor bn_42_mean = DeviceTensor(512);
    DeviceTensor bn_42_var = DeviceTensor(512);
    DeviceTensor bn_42_weight = DeviceTensor(512);
    DeviceTensor bn_42_bias = DeviceTensor(512);
    DeviceTensor conv_44_weight = DeviceTensor(131072);
    DeviceTensor bn_44_mean = DeviceTensor(256);
    DeviceTensor bn_44_var = DeviceTensor(256);
    DeviceTensor bn_44_weight = DeviceTensor(256);
    DeviceTensor bn_44_bias = DeviceTensor(256);
    DeviceTensor conv_45_weight = DeviceTensor(1179648);
    DeviceTensor bn_45_mean = DeviceTensor(512);
    DeviceTensor bn_45_var = DeviceTensor(512);
    DeviceTensor bn_45_weight = DeviceTensor(512);
    DeviceTensor bn_45_bias = DeviceTensor(512);
    DeviceTensor conv_47_weight = DeviceTensor(131072);
    DeviceTensor bn_47_mean = DeviceTensor(256);
    DeviceTensor bn_47_var = DeviceTensor(256);
    DeviceTensor bn_47_weight = DeviceTensor(256);
    DeviceTensor bn_47_bias = DeviceTensor(256);
    DeviceTensor conv_48_weight = DeviceTensor(1179648);
    DeviceTensor bn_48_mean = DeviceTensor(512);
    DeviceTensor bn_48_var = DeviceTensor(512);
    DeviceTensor bn_48_weight = DeviceTensor(512);
    DeviceTensor bn_48_bias = DeviceTensor(512);
    DeviceTensor conv_50_weight = DeviceTensor(131072);
    DeviceTensor bn_50_mean = DeviceTensor(256);
    DeviceTensor bn_50_var = DeviceTensor(256);
    DeviceTensor bn_50_weight = DeviceTensor(256);
    DeviceTensor bn_50_bias = DeviceTensor(256);
    DeviceTensor conv_51_weight = DeviceTensor(1179648);
    DeviceTensor bn_51_mean = DeviceTensor(512);
    DeviceTensor bn_51_var = DeviceTensor(512);
    DeviceTensor bn_51_weight = DeviceTensor(512);
    DeviceTensor bn_51_bias = DeviceTensor(512);
    DeviceTensor conv_53_weight = DeviceTensor(131072);
    DeviceTensor bn_53_mean = DeviceTensor(256);
    DeviceTensor bn_53_var = DeviceTensor(256);
    DeviceTensor bn_53_weight = DeviceTensor(256);
    DeviceTensor bn_53_bias = DeviceTensor(256);
    DeviceTensor conv_54_weight = DeviceTensor(1179648);
    DeviceTensor bn_54_mean = DeviceTensor(512);
    DeviceTensor bn_54_var = DeviceTensor(512);
    DeviceTensor bn_54_weight = DeviceTensor(512);
    DeviceTensor bn_54_bias = DeviceTensor(512);
    DeviceTensor conv_56_weight = DeviceTensor(131072);
    DeviceTensor bn_56_mean = DeviceTensor(256);
    DeviceTensor bn_56_var = DeviceTensor(256);
    DeviceTensor bn_56_weight = DeviceTensor(256);
    DeviceTensor bn_56_bias = DeviceTensor(256);
    DeviceTensor conv_57_weight = DeviceTensor(1179648);
    DeviceTensor bn_57_mean = DeviceTensor(512);
    DeviceTensor bn_57_var = DeviceTensor(512);
    DeviceTensor bn_57_weight = DeviceTensor(512);
    DeviceTensor bn_57_bias = DeviceTensor(512);
    DeviceTensor conv_59_weight = DeviceTensor(131072);
    DeviceTensor bn_59_mean = DeviceTensor(256);
    DeviceTensor bn_59_var = DeviceTensor(256);
    DeviceTensor bn_59_weight = DeviceTensor(256);
    DeviceTensor bn_59_bias = DeviceTensor(256);
    DeviceTensor conv_60_weight = DeviceTensor(1179648);
    DeviceTensor bn_60_mean = DeviceTensor(512);
    DeviceTensor bn_60_var = DeviceTensor(512);
    DeviceTensor bn_60_weight = DeviceTensor(512);
    DeviceTensor bn_60_bias = DeviceTensor(512);
    DeviceTensor conv_62_weight = DeviceTensor(4718592);
    DeviceTensor bn_62_mean = DeviceTensor(1024);
    DeviceTensor bn_62_var = DeviceTensor(1024);
    DeviceTensor bn_62_weight = DeviceTensor(1024);
    DeviceTensor bn_62_bias = DeviceTensor(1024);
    DeviceTensor conv_63_weight = DeviceTensor(524288);
    DeviceTensor bn_63_mean = DeviceTensor(512);
    DeviceTensor bn_63_var = DeviceTensor(512);
    DeviceTensor bn_63_weight = DeviceTensor(512);
    DeviceTensor bn_63_bias = DeviceTensor(512);
    DeviceTensor conv_64_weight = DeviceTensor(4718592);
    DeviceTensor bn_64_mean = DeviceTensor(1024);
    DeviceTensor bn_64_var = DeviceTensor(1024);
    DeviceTensor bn_64_weight = DeviceTensor(1024);
    DeviceTensor bn_64_bias = DeviceTensor(1024);
    DeviceTensor conv_66_weight = DeviceTensor(524288);
    DeviceTensor bn_66_mean = DeviceTensor(512);
    DeviceTensor bn_66_var = DeviceTensor(512);
    DeviceTensor bn_66_weight = DeviceTensor(512);
    DeviceTensor bn_66_bias = DeviceTensor(512);
    DeviceTensor conv_67_weight = DeviceTensor(4718592);
    DeviceTensor bn_67_mean = DeviceTensor(1024);
    DeviceTensor bn_67_var = DeviceTensor(1024);
    DeviceTensor bn_67_weight = DeviceTensor(1024);
    DeviceTensor bn_67_bias = DeviceTensor(1024);
    DeviceTensor conv_69_weight = DeviceTensor(524288);
    DeviceTensor bn_69_mean = DeviceTensor(512);
    DeviceTensor bn_69_var = DeviceTensor(512);
    DeviceTensor bn_69_weight = DeviceTensor(512);
    DeviceTensor bn_69_bias = DeviceTensor(512);
    DeviceTensor conv_70_weight = DeviceTensor(4718592);
    DeviceTensor bn_70_mean = DeviceTensor(1024);
    DeviceTensor bn_70_var = DeviceTensor(1024);
    DeviceTensor bn_70_weight = DeviceTensor(1024);
    DeviceTensor bn_70_bias = DeviceTensor(1024);
    DeviceTensor conv_72_weight = DeviceTensor(524288);
    DeviceTensor bn_72_mean = DeviceTensor(512);
    DeviceTensor bn_72_var = DeviceTensor(512);
    DeviceTensor bn_72_weight = DeviceTensor(512);
    DeviceTensor bn_72_bias = DeviceTensor(512);
    DeviceTensor conv_73_weight = DeviceTensor(4718592);
    DeviceTensor bn_73_mean = DeviceTensor(1024);
    DeviceTensor bn_73_var = DeviceTensor(1024);
    DeviceTensor bn_73_weight = DeviceTensor(1024);
    DeviceTensor bn_73_bias = DeviceTensor(1024);
    DeviceTensor conv_75_weight = DeviceTensor(524288);
    DeviceTensor bn_75_mean = DeviceTensor(512);
    DeviceTensor bn_75_var = DeviceTensor(512);
    DeviceTensor bn_75_weight = DeviceTensor(512);
    DeviceTensor bn_75_bias = DeviceTensor(512);
    DeviceTensor conv_76_weight = DeviceTensor(4718592);
    DeviceTensor bn_76_mean = DeviceTensor(1024);
    DeviceTensor bn_76_var = DeviceTensor(1024);
    DeviceTensor bn_76_weight = DeviceTensor(1024);
    DeviceTensor bn_76_bias = DeviceTensor(1024);
    DeviceTensor conv_77_weight = DeviceTensor(524288);
    DeviceTensor bn_77_mean = DeviceTensor(512);
    DeviceTensor bn_77_var = DeviceTensor(512);
    DeviceTensor bn_77_weight = DeviceTensor(512);
    DeviceTensor bn_77_bias = DeviceTensor(512);
    DeviceTensor conv_78_weight = DeviceTensor(4718592);
    DeviceTensor bn_78_mean = DeviceTensor(1024);
    DeviceTensor bn_78_var = DeviceTensor(1024);
    DeviceTensor bn_78_weight = DeviceTensor(1024);
    DeviceTensor bn_78_bias = DeviceTensor(1024);
    DeviceTensor conv_79_weight = DeviceTensor(524288);
    DeviceTensor bn_79_mean = DeviceTensor(512);
    DeviceTensor bn_79_var = DeviceTensor(512);
    DeviceTensor bn_79_weight = DeviceTensor(512);
    DeviceTensor bn_79_bias = DeviceTensor(512);
    DeviceTensor conv_80_weight = DeviceTensor(4718592);
    DeviceTensor bn_80_mean = DeviceTensor(1024);
    DeviceTensor bn_80_var = DeviceTensor(1024);
    DeviceTensor bn_80_weight = DeviceTensor(1024);
    DeviceTensor bn_80_bias = DeviceTensor(1024);
    DeviceTensor conv_81_bias = DeviceTensor(255);
    DeviceTensor conv_81_weight = DeviceTensor(261120);
    DeviceTensor conv_84_weight = DeviceTensor(131072);
    DeviceTensor bn_84_mean = DeviceTensor(256);
    DeviceTensor bn_84_var = DeviceTensor(256);
    DeviceTensor bn_84_weight = DeviceTensor(256);
    DeviceTensor bn_84_bias = DeviceTensor(256);
    DeviceTensor conv_87_weight = DeviceTensor(196608);
    DeviceTensor bn_87_mean = DeviceTensor(256);
    DeviceTensor bn_87_var = DeviceTensor(256);
    DeviceTensor bn_87_weight = DeviceTensor(256);
    DeviceTensor bn_87_bias = DeviceTensor(256);
    DeviceTensor conv_88_weight = DeviceTensor(1179648);
    DeviceTensor bn_88_mean = DeviceTensor(512);
    DeviceTensor bn_88_var = DeviceTensor(512);
    DeviceTensor bn_88_weight = DeviceTensor(512);
    DeviceTensor bn_88_bias = DeviceTensor(512);
    DeviceTensor conv_89_weight = DeviceTensor(131072);
    DeviceTensor bn_89_mean = DeviceTensor(256);
    DeviceTensor bn_89_var = DeviceTensor(256);
    DeviceTensor bn_89_weight = DeviceTensor(256);
    DeviceTensor bn_89_bias = DeviceTensor(256);
    DeviceTensor conv_90_weight = DeviceTensor(1179648);
    DeviceTensor bn_90_mean = DeviceTensor(512);
    DeviceTensor bn_90_var = DeviceTensor(512);
    DeviceTensor bn_90_weight = DeviceTensor(512);
    DeviceTensor bn_90_bias = DeviceTensor(512);
    DeviceTensor conv_91_weight = DeviceTensor(131072);
    DeviceTensor bn_91_mean = DeviceTensor(256);
    DeviceTensor bn_91_var = DeviceTensor(256);
    DeviceTensor bn_91_weight = DeviceTensor(256);
    DeviceTensor bn_91_bias = DeviceTensor(256);
    DeviceTensor conv_92_weight = DeviceTensor(1179648);
    DeviceTensor bn_92_mean = DeviceTensor(512);
    DeviceTensor bn_92_var = DeviceTensor(512);
    DeviceTensor bn_92_weight = DeviceTensor(512);
    DeviceTensor bn_92_bias = DeviceTensor(512);
    DeviceTensor conv_93_bias = DeviceTensor(255);
    DeviceTensor conv_93_weight = DeviceTensor(130560);
    DeviceTensor conv_96_weight = DeviceTensor(32768);
    DeviceTensor bn_96_mean = DeviceTensor(128);
    DeviceTensor bn_96_var = DeviceTensor(128);
    DeviceTensor bn_96_weight = DeviceTensor(128);
    DeviceTensor bn_96_bias = DeviceTensor(128);
    DeviceTensor conv_99_weight = DeviceTensor(49152);
    DeviceTensor bn_99_mean = DeviceTensor(128);
    DeviceTensor bn_99_var = DeviceTensor(128);
    DeviceTensor bn_99_weight = DeviceTensor(128);
    DeviceTensor bn_99_bias = DeviceTensor(128);
    DeviceTensor conv_100_weight = DeviceTensor(294912);
    DeviceTensor bn_100_mean = DeviceTensor(256);
    DeviceTensor bn_100_var = DeviceTensor(256);
    DeviceTensor bn_100_weight = DeviceTensor(256);
    DeviceTensor bn_100_bias = DeviceTensor(256);
    DeviceTensor conv_101_weight = DeviceTensor(32768);
    DeviceTensor bn_101_mean = DeviceTensor(128);
    DeviceTensor bn_101_var = DeviceTensor(128);
    DeviceTensor bn_101_weight = DeviceTensor(128);
    DeviceTensor bn_101_bias = DeviceTensor(128);
    DeviceTensor conv_102_weight = DeviceTensor(294912);
    DeviceTensor bn_102_mean = DeviceTensor(256);
    DeviceTensor bn_102_var = DeviceTensor(256);
    DeviceTensor bn_102_weight = DeviceTensor(256);
    DeviceTensor bn_102_bias = DeviceTensor(256);
    DeviceTensor conv_103_weight = DeviceTensor(32768);
    DeviceTensor bn_103_mean = DeviceTensor(128);
    DeviceTensor bn_103_var = DeviceTensor(128);
    DeviceTensor bn_103_weight = DeviceTensor(128);
    DeviceTensor bn_103_bias = DeviceTensor(128);
    DeviceTensor conv_104_weight = DeviceTensor(294912);
    DeviceTensor bn_104_mean = DeviceTensor(256);
    DeviceTensor bn_104_var = DeviceTensor(256);
    DeviceTensor bn_104_weight = DeviceTensor(256);
    DeviceTensor bn_104_bias = DeviceTensor(256);
    DeviceTensor conv_105_bias = DeviceTensor(255);
    DeviceTensor conv_105_weight = DeviceTensor(65280);




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