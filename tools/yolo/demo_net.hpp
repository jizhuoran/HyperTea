#include "hypertea/hypertea.hpp"
#include "kernels/conv_kernel.cl"

namespace hypertea {


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    
    int batch_size = prediction.size(0)
    int stride =  inp_dim / prediction.size(2)
    int grid_size = inp_dim / stride
    int bbox_attrs = 5 + num_classes
    int num_anchors = anchors.size() / 2;
    
    // the batch_size is 1 stride is 32 grid_size is 13 num_class is 80
    
    for (int i = 0; i < anchors.size(); ++i) {
        anchors[i] /= stride;
    }


    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)


    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    

    
    #Add the center offsets
    grid_len = np.arange(grid_size)
    a,b = np.meshgrid(grid_len, grid_len)
    
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    

    
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    
    prediction[:,:,:2] += x_y_offset
      
    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)
    

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    #Softmax the class scores
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride
   
    
    return prediction


// template <typename DeviceTensor>
// class Yolo_op {

// public:
//     explicit Yolo_op(std::vector<int> anchors, int in_dim, int class_num) 
//     : anchors_(anchors), in_dim_(in_dim), class_num_(class_num) {}
    
//     virtual inline const char* type() const override { return "Yolo_op"; }
//     DeviceTensor& operator()(DeviceTensor& input) {

//         return input;
//     }

// private:
//     std::vector<int> anchors_;
//     int in_dim_;
//     int class_num_;
// };

// void yolo_layer(DeviceTensor& x) {

//     x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
    
//     if type(x) == int:
//         continue

    
//     if not write:
//         detections = x
//         write = 1
    
//     else:
//         detections = torch.cat((detections, x), 1)
    
//     outputs[i] = outputs[i-1] 
// }




template <typename DeviceTensor>
class yolo_net {

public:

    yolo_net(const std::string &param_file) { 

        compile_opencl_kernels(conv_opencl_funcs, " ");
        
        load_weight_to_tensor(param_file, param);

    }

    void inference( std::vector<float> &data_from_user, std::vector<int> &data_to_user) {
        
        DeviceTensor data(data_from_user);

        auto x0 = (conv_0(batch_norm_0(leaky_0(data))));
        auto x1 = (conv_1(batch_norm_1(leaky_1(x0))));
        auto x2 = (conv_2(batch_norm_2(leaky_2(x1))));
        auto x3 = (conv_3(batch_norm_3(leaky_3(x2))));

        auto x4 = x1 + x3;
        auto x5 = (conv_5(batch_norm_5(leaky_5(x4))));
        auto x6 = (conv_6(batch_norm_6(leaky_6(x5))));
        auto x7 = (conv_7(batch_norm_7(leaky_7(x6))));
        auto x8 = x5 + x7;
        auto x9 = (conv_9(batch_norm_9(leaky_9(x8))));
        auto x10 = (conv_10(batch_norm_10(leaky_10(x9))));
        auto x11 = x8 + x10;
        auto x12 = (conv_12(batch_norm_12(leaky_12(x11))));
        auto x13 = (conv_13(batch_norm_13(leaky_13(x12))));
        auto x14 = (conv_14(batch_norm_14(leaky_14(x13))));
        auto x15 = x12 + x14;
        auto x16 = (conv_16(batch_norm_16(leaky_16(x15))));
        auto x17 = (conv_17(batch_norm_17(leaky_17(x16))));
        auto x18 = x15 + x17;
        auto x19 = (conv_19(batch_norm_19(leaky_19(x18))));
        auto x20 = (conv_20(batch_norm_20(leaky_20(x19))));
        auto x21 = x18 + x20;
        auto x22 = (conv_22(batch_norm_22(leaky_22(x21))));
        auto x23 = (conv_23(batch_norm_23(leaky_23(x22))));
        auto x24 = x21 + x23;
        auto x25 = (conv_25(batch_norm_25(leaky_25(x24))));
        auto x26 = (conv_26(batch_norm_26(leaky_26(x25))));
        auto x27 = x24 + x26;
        auto x28 = (conv_28(batch_norm_28(leaky_28(x27))));
        auto x29 = (conv_29(batch_norm_29(leaky_29(x28))));
        auto x30 = x27 + x29;
        auto x31 = (conv_31(batch_norm_31(leaky_31(x30))));
        auto x32 = (conv_32(batch_norm_32(leaky_32(x31))));
        auto x33 = x30 + x32;
        auto x34 = (conv_34(batch_norm_34(leaky_34(x33))));
        auto x35 = (conv_35(batch_norm_35(leaky_35(x34))));
        auto x36 = x33 + x35;
        auto x37 = (conv_37(batch_norm_37(leaky_37(x36))));
        auto x38 = (conv_38(batch_norm_38(leaky_38(x37))));
        auto x39 = (conv_39(batch_norm_39(leaky_39(x38))));
        auto x40 = x37 + x39;
        auto x41 = (conv_41(batch_norm_41(leaky_41(x40))));
        auto x42 = (conv_42(batch_norm_42(leaky_42(x41))));
        auto x43 = x40 + x42;
        auto x44 = (conv_44(batch_norm_44(leaky_44(x43))));
        auto x45 = (conv_45(batch_norm_45(leaky_45(x44))));
        auto x46 = x43 + x45;
        auto x47 = (conv_47(batch_norm_47(leaky_47(x46))));
        auto x48 = (conv_48(batch_norm_48(leaky_48(x47))));
        auto x49 = x46 + x48;
        auto x50 = (conv_50(batch_norm_50(leaky_50(x49))));
        auto x51 = (conv_51(batch_norm_51(leaky_51(x50))));
        auto x52 = x49 + x51;
        auto x53 = (conv_53(batch_norm_53(leaky_53(x52))));
        auto x54 = (conv_54(batch_norm_54(leaky_54(x53))));
        auto x55 = x52 + x54;
        auto x56 = (conv_56(batch_norm_56(leaky_56(x55))));
        auto x57 = (conv_57(batch_norm_57(leaky_57(x56))));
        auto x58 = x55 + x57;
        auto x59 = (conv_59(batch_norm_59(leaky_59(x58))));
        auto x60 = (conv_60(batch_norm_60(leaky_60(x59))));
        auto x61 = x58 + x60;
        auto x62 = (conv_62(batch_norm_62(leaky_62(x61))));
        auto x63 = (conv_63(batch_norm_63(leaky_63(x62))));
        auto x64 = (conv_64(batch_norm_64(leaky_64(x63))));
        auto x65 = x62 + x64;
        auto x66 = (conv_66(batch_norm_66(leaky_66(x65))));
        auto x67 = (conv_67(batch_norm_67(leaky_67(x66))));
        auto x68 = x65 + x67;
        auto x69 = (conv_69(batch_norm_69(leaky_69(x68))));
        auto x70 = (conv_70(batch_norm_70(leaky_70(x69))));
        auto x71 = x68 + x70;
        auto x72 = (conv_72(batch_norm_72(leaky_72(x71))));
        auto x73 = (conv_73(batch_norm_73(leaky_73(x72))));
        auto x74 = x71 + x73;
        auto x75 = (conv_75(batch_norm_75(leaky_75(x74))));
        auto x76 = (conv_76(batch_norm_76(leaky_76(x75))));
        auto x77 = (conv_77(batch_norm_77(leaky_77(x76))));
        auto x78 = (conv_78(batch_norm_78(leaky_78(x77))));
        auto x79 = (conv_79(batch_norm_79(leaky_79(x78))));
        auto x80 = (conv_80(batch_norm_80(leaky_80(x79))));
        auto x81 = conv_81(x80);

        std::cout << "The count is " << x81.count() << std::endl;

        auto x82 = x81;    //x82 is a yolo layer
        auto x83 = x79;
        auto x84 = (conv_84(batch_norm_84(leaky_84(x83))));
            
        auto x85 = upsampling_85(x84);

        auto x86 = x61 + x85;
        auto x87 = (conv_87(batch_norm_87(leaky_87(x86))));
        auto x88 = (conv_88(batch_norm_88(leaky_88(x87))));
        auto x89 = (conv_89(batch_norm_89(leaky_89(x88))));
        auto x90 = (conv_90(batch_norm_90(leaky_90(x89))));
        auto x91 = (conv_91(batch_norm_91(leaky_91(x90))));
        auto x92 = (conv_92(batch_norm_92(leaky_92(x91))));
        auto x93 = conv_93(x92);
        auto x94 = x93;//    x94 is a yolo layer
        auto x95 = x91;
        auto x96 = (conv_96(batch_norm_96(leaky_96(x95))));
            
        auto x97 = upsampling_97(x96);

        auto x98 = x36 + x97;
        auto x99 = (conv_99(batch_norm_99(leaky_99(x98))));
        auto x100 = (conv_100(batch_norm_100(leaky_100(x99))));
        auto x101 = (conv_101(batch_norm_101(leaky_101(x100))));
        auto x102 = (conv_102(batch_norm_102(leaky_102(x101))));
        auto x103 = (conv_103(batch_norm_103(leaky_103(x102))));
        auto x104 = (conv_104(batch_norm_104(leaky_104(x103))));
        auto x105 = (conv_105(x104));
        auto x106 = x105;//    x106 is a yolo layer






        
        
    }

private:
    
    DeviceTensor param = DeviceTensor(62001757);

     DeviceTensor conv_0_weight = param.sub_view(0, 864);
     DeviceTensor batch_norm_0_mean = param.sub_view(864, 32);
     DeviceTensor batch_norm_0_var = param.sub_view(896, 32);
     DeviceTensor batch_norm_0_weight = param.sub_view(928, 32);
     DeviceTensor batch_norm_0_bias = param.sub_view(960, 32);
     DeviceTensor conv_1_weight = param.sub_view(992, 18432);
     DeviceTensor batch_norm_1_mean = param.sub_view(19424, 64);
     DeviceTensor batch_norm_1_var = param.sub_view(19488, 64);
     DeviceTensor batch_norm_1_weight = param.sub_view(19552, 64);
     DeviceTensor batch_norm_1_bias = param.sub_view(19616, 64);
     DeviceTensor conv_2_weight = param.sub_view(19680, 2048);
     DeviceTensor batch_norm_2_mean = param.sub_view(21728, 32);
     DeviceTensor batch_norm_2_var = param.sub_view(21760, 32);
     DeviceTensor batch_norm_2_weight = param.sub_view(21792, 32);
     DeviceTensor batch_norm_2_bias = param.sub_view(21824, 32);
     DeviceTensor conv_3_weight = param.sub_view(21856, 18432);
     DeviceTensor batch_norm_3_mean = param.sub_view(40288, 64);
     DeviceTensor batch_norm_3_var = param.sub_view(40352, 64);
     DeviceTensor batch_norm_3_weight = param.sub_view(40416, 64);
     DeviceTensor batch_norm_3_bias = param.sub_view(40480, 64);
     DeviceTensor conv_5_weight = param.sub_view(40544, 73728);
     DeviceTensor batch_norm_5_mean = param.sub_view(114272, 128);
     DeviceTensor batch_norm_5_var = param.sub_view(114400, 128);
     DeviceTensor batch_norm_5_weight = param.sub_view(114528, 128);
     DeviceTensor batch_norm_5_bias = param.sub_view(114656, 128);
     DeviceTensor conv_6_weight = param.sub_view(114784, 8192);
     DeviceTensor batch_norm_6_mean = param.sub_view(122976, 64);
     DeviceTensor batch_norm_6_var = param.sub_view(123040, 64);
     DeviceTensor batch_norm_6_weight = param.sub_view(123104, 64);
     DeviceTensor batch_norm_6_bias = param.sub_view(123168, 64);
     DeviceTensor conv_7_weight = param.sub_view(123232, 73728);
     DeviceTensor batch_norm_7_mean = param.sub_view(196960, 128);
     DeviceTensor batch_norm_7_var = param.sub_view(197088, 128);
     DeviceTensor batch_norm_7_weight = param.sub_view(197216, 128);
     DeviceTensor batch_norm_7_bias = param.sub_view(197344, 128);
     DeviceTensor conv_9_weight = param.sub_view(197472, 8192);
     DeviceTensor batch_norm_9_mean = param.sub_view(205664, 64);
     DeviceTensor batch_norm_9_var = param.sub_view(205728, 64);
     DeviceTensor batch_norm_9_weight = param.sub_view(205792, 64);
     DeviceTensor batch_norm_9_bias = param.sub_view(205856, 64);
     DeviceTensor conv_10_weight = param.sub_view(205920, 73728);
     DeviceTensor batch_norm_10_mean = param.sub_view(279648, 128);
     DeviceTensor batch_norm_10_var = param.sub_view(279776, 128);
     DeviceTensor batch_norm_10_weight = param.sub_view(279904, 128);
     DeviceTensor batch_norm_10_bias = param.sub_view(280032, 128);
     DeviceTensor conv_12_weight = param.sub_view(280160, 294912);
     DeviceTensor batch_norm_12_mean = param.sub_view(575072, 256);
     DeviceTensor batch_norm_12_var = param.sub_view(575328, 256);
     DeviceTensor batch_norm_12_weight = param.sub_view(575584, 256);
     DeviceTensor batch_norm_12_bias = param.sub_view(575840, 256);
     DeviceTensor conv_13_weight = param.sub_view(576096, 32768);
     DeviceTensor batch_norm_13_mean = param.sub_view(608864, 128);
     DeviceTensor batch_norm_13_var = param.sub_view(608992, 128);
     DeviceTensor batch_norm_13_weight = param.sub_view(609120, 128);
     DeviceTensor batch_norm_13_bias = param.sub_view(609248, 128);
     DeviceTensor conv_14_weight = param.sub_view(609376, 294912);
     DeviceTensor batch_norm_14_mean = param.sub_view(904288, 256);
     DeviceTensor batch_norm_14_var = param.sub_view(904544, 256);
     DeviceTensor batch_norm_14_weight = param.sub_view(904800, 256);
     DeviceTensor batch_norm_14_bias = param.sub_view(905056, 256);
     DeviceTensor conv_16_weight = param.sub_view(905312, 32768);
     DeviceTensor batch_norm_16_mean = param.sub_view(938080, 128);
     DeviceTensor batch_norm_16_var = param.sub_view(938208, 128);
     DeviceTensor batch_norm_16_weight = param.sub_view(938336, 128);
     DeviceTensor batch_norm_16_bias = param.sub_view(938464, 128);
     DeviceTensor conv_17_weight = param.sub_view(938592, 294912);
     DeviceTensor batch_norm_17_mean = param.sub_view(1233504, 256);
     DeviceTensor batch_norm_17_var = param.sub_view(1233760, 256);
     DeviceTensor batch_norm_17_weight = param.sub_view(1234016, 256);
     DeviceTensor batch_norm_17_bias = param.sub_view(1234272, 256);
     DeviceTensor conv_19_weight = param.sub_view(1234528, 32768);
     DeviceTensor batch_norm_19_mean = param.sub_view(1267296, 128);
     DeviceTensor batch_norm_19_var = param.sub_view(1267424, 128);
     DeviceTensor batch_norm_19_weight = param.sub_view(1267552, 128);
     DeviceTensor batch_norm_19_bias = param.sub_view(1267680, 128);
     DeviceTensor conv_20_weight = param.sub_view(1267808, 294912);
     DeviceTensor batch_norm_20_mean = param.sub_view(1562720, 256);
     DeviceTensor batch_norm_20_var = param.sub_view(1562976, 256);
     DeviceTensor batch_norm_20_weight = param.sub_view(1563232, 256);
     DeviceTensor batch_norm_20_bias = param.sub_view(1563488, 256);
     DeviceTensor conv_22_weight = param.sub_view(1563744, 32768);
     DeviceTensor batch_norm_22_mean = param.sub_view(1596512, 128);
     DeviceTensor batch_norm_22_var = param.sub_view(1596640, 128);
     DeviceTensor batch_norm_22_weight = param.sub_view(1596768, 128);
     DeviceTensor batch_norm_22_bias = param.sub_view(1596896, 128);
     DeviceTensor conv_23_weight = param.sub_view(1597024, 294912);
     DeviceTensor batch_norm_23_mean = param.sub_view(1891936, 256);
     DeviceTensor batch_norm_23_var = param.sub_view(1892192, 256);
     DeviceTensor batch_norm_23_weight = param.sub_view(1892448, 256);
     DeviceTensor batch_norm_23_bias = param.sub_view(1892704, 256);
     DeviceTensor conv_25_weight = param.sub_view(1892960, 32768);
     DeviceTensor batch_norm_25_mean = param.sub_view(1925728, 128);
     DeviceTensor batch_norm_25_var = param.sub_view(1925856, 128);
     DeviceTensor batch_norm_25_weight = param.sub_view(1925984, 128);
     DeviceTensor batch_norm_25_bias = param.sub_view(1926112, 128);
     DeviceTensor conv_26_weight = param.sub_view(1926240, 294912);
     DeviceTensor batch_norm_26_mean = param.sub_view(2221152, 256);
     DeviceTensor batch_norm_26_var = param.sub_view(2221408, 256);
     DeviceTensor batch_norm_26_weight = param.sub_view(2221664, 256);
     DeviceTensor batch_norm_26_bias = param.sub_view(2221920, 256);
     DeviceTensor conv_28_weight = param.sub_view(2222176, 32768);
     DeviceTensor batch_norm_28_mean = param.sub_view(2254944, 128);
     DeviceTensor batch_norm_28_var = param.sub_view(2255072, 128);
     DeviceTensor batch_norm_28_weight = param.sub_view(2255200, 128);
     DeviceTensor batch_norm_28_bias = param.sub_view(2255328, 128);
     DeviceTensor conv_29_weight = param.sub_view(2255456, 294912);
     DeviceTensor batch_norm_29_mean = param.sub_view(2550368, 256);
     DeviceTensor batch_norm_29_var = param.sub_view(2550624, 256);
     DeviceTensor batch_norm_29_weight = param.sub_view(2550880, 256);
     DeviceTensor batch_norm_29_bias = param.sub_view(2551136, 256);
     DeviceTensor conv_31_weight = param.sub_view(2551392, 32768);
     DeviceTensor batch_norm_31_mean = param.sub_view(2584160, 128);
     DeviceTensor batch_norm_31_var = param.sub_view(2584288, 128);
     DeviceTensor batch_norm_31_weight = param.sub_view(2584416, 128);
     DeviceTensor batch_norm_31_bias = param.sub_view(2584544, 128);
     DeviceTensor conv_32_weight = param.sub_view(2584672, 294912);
     DeviceTensor batch_norm_32_mean = param.sub_view(2879584, 256);
     DeviceTensor batch_norm_32_var = param.sub_view(2879840, 256);
     DeviceTensor batch_norm_32_weight = param.sub_view(2880096, 256);
     DeviceTensor batch_norm_32_bias = param.sub_view(2880352, 256);
     DeviceTensor conv_34_weight = param.sub_view(2880608, 32768);
     DeviceTensor batch_norm_34_mean = param.sub_view(2913376, 128);
     DeviceTensor batch_norm_34_var = param.sub_view(2913504, 128);
     DeviceTensor batch_norm_34_weight = param.sub_view(2913632, 128);
     DeviceTensor batch_norm_34_bias = param.sub_view(2913760, 128);
     DeviceTensor conv_35_weight = param.sub_view(2913888, 294912);
     DeviceTensor batch_norm_35_mean = param.sub_view(3208800, 256);
     DeviceTensor batch_norm_35_var = param.sub_view(3209056, 256);
     DeviceTensor batch_norm_35_weight = param.sub_view(3209312, 256);
     DeviceTensor batch_norm_35_bias = param.sub_view(3209568, 256);
     DeviceTensor conv_37_weight = param.sub_view(3209824, 1179648);
     DeviceTensor batch_norm_37_mean = param.sub_view(4389472, 512);
     DeviceTensor batch_norm_37_var = param.sub_view(4389984, 512);
     DeviceTensor batch_norm_37_weight = param.sub_view(4390496, 512);
     DeviceTensor batch_norm_37_bias = param.sub_view(4391008, 512);
     DeviceTensor conv_38_weight = param.sub_view(4391520, 131072);
     DeviceTensor batch_norm_38_mean = param.sub_view(4522592, 256);
     DeviceTensor batch_norm_38_var = param.sub_view(4522848, 256);
     DeviceTensor batch_norm_38_weight = param.sub_view(4523104, 256);
     DeviceTensor batch_norm_38_bias = param.sub_view(4523360, 256);
     DeviceTensor conv_39_weight = param.sub_view(4523616, 1179648);
     DeviceTensor batch_norm_39_mean = param.sub_view(5703264, 512);
     DeviceTensor batch_norm_39_var = param.sub_view(5703776, 512);
     DeviceTensor batch_norm_39_weight = param.sub_view(5704288, 512);
     DeviceTensor batch_norm_39_bias = param.sub_view(5704800, 512);
     DeviceTensor conv_41_weight = param.sub_view(5705312, 131072);
     DeviceTensor batch_norm_41_mean = param.sub_view(5836384, 256);
     DeviceTensor batch_norm_41_var = param.sub_view(5836640, 256);
     DeviceTensor batch_norm_41_weight = param.sub_view(5836896, 256);
     DeviceTensor batch_norm_41_bias = param.sub_view(5837152, 256);
     DeviceTensor conv_42_weight = param.sub_view(5837408, 1179648);
     DeviceTensor batch_norm_42_mean = param.sub_view(7017056, 512);
     DeviceTensor batch_norm_42_var = param.sub_view(7017568, 512);
     DeviceTensor batch_norm_42_weight = param.sub_view(7018080, 512);
     DeviceTensor batch_norm_42_bias = param.sub_view(7018592, 512);
     DeviceTensor conv_44_weight = param.sub_view(7019104, 131072);
     DeviceTensor batch_norm_44_mean = param.sub_view(7150176, 256);
     DeviceTensor batch_norm_44_var = param.sub_view(7150432, 256);
     DeviceTensor batch_norm_44_weight = param.sub_view(7150688, 256);
     DeviceTensor batch_norm_44_bias = param.sub_view(7150944, 256);
     DeviceTensor conv_45_weight = param.sub_view(7151200, 1179648);
     DeviceTensor batch_norm_45_mean = param.sub_view(8330848, 512);
     DeviceTensor batch_norm_45_var = param.sub_view(8331360, 512);
     DeviceTensor batch_norm_45_weight = param.sub_view(8331872, 512);
     DeviceTensor batch_norm_45_bias = param.sub_view(8332384, 512);
     DeviceTensor conv_47_weight = param.sub_view(8332896, 131072);
     DeviceTensor batch_norm_47_mean = param.sub_view(8463968, 256);
     DeviceTensor batch_norm_47_var = param.sub_view(8464224, 256);
     DeviceTensor batch_norm_47_weight = param.sub_view(8464480, 256);
     DeviceTensor batch_norm_47_bias = param.sub_view(8464736, 256);
     DeviceTensor conv_48_weight = param.sub_view(8464992, 1179648);
     DeviceTensor batch_norm_48_mean = param.sub_view(9644640, 512);
     DeviceTensor batch_norm_48_var = param.sub_view(9645152, 512);
     DeviceTensor batch_norm_48_weight = param.sub_view(9645664, 512);
     DeviceTensor batch_norm_48_bias = param.sub_view(9646176, 512);
     DeviceTensor conv_50_weight = param.sub_view(9646688, 131072);
     DeviceTensor batch_norm_50_mean = param.sub_view(9777760, 256);
     DeviceTensor batch_norm_50_var = param.sub_view(9778016, 256);
     DeviceTensor batch_norm_50_weight = param.sub_view(9778272, 256);
     DeviceTensor batch_norm_50_bias = param.sub_view(9778528, 256);
     DeviceTensor conv_51_weight = param.sub_view(9778784, 1179648);
     DeviceTensor batch_norm_51_mean = param.sub_view(10958432, 512);
     DeviceTensor batch_norm_51_var = param.sub_view(10958944, 512);
     DeviceTensor batch_norm_51_weight = param.sub_view(10959456, 512);
     DeviceTensor batch_norm_51_bias = param.sub_view(10959968, 512);
     DeviceTensor conv_53_weight = param.sub_view(10960480, 131072);
     DeviceTensor batch_norm_53_mean = param.sub_view(11091552, 256);
     DeviceTensor batch_norm_53_var = param.sub_view(11091808, 256);
     DeviceTensor batch_norm_53_weight = param.sub_view(11092064, 256);
     DeviceTensor batch_norm_53_bias = param.sub_view(11092320, 256);
     DeviceTensor conv_54_weight = param.sub_view(11092576, 1179648);
     DeviceTensor batch_norm_54_mean = param.sub_view(12272224, 512);
     DeviceTensor batch_norm_54_var = param.sub_view(12272736, 512);
     DeviceTensor batch_norm_54_weight = param.sub_view(12273248, 512);
     DeviceTensor batch_norm_54_bias = param.sub_view(12273760, 512);
     DeviceTensor conv_56_weight = param.sub_view(12274272, 131072);
     DeviceTensor batch_norm_56_mean = param.sub_view(12405344, 256);
     DeviceTensor batch_norm_56_var = param.sub_view(12405600, 256);
     DeviceTensor batch_norm_56_weight = param.sub_view(12405856, 256);
     DeviceTensor batch_norm_56_bias = param.sub_view(12406112, 256);
     DeviceTensor conv_57_weight = param.sub_view(12406368, 1179648);
     DeviceTensor batch_norm_57_mean = param.sub_view(13586016, 512);
     DeviceTensor batch_norm_57_var = param.sub_view(13586528, 512);
     DeviceTensor batch_norm_57_weight = param.sub_view(13587040, 512);
     DeviceTensor batch_norm_57_bias = param.sub_view(13587552, 512);
     DeviceTensor conv_59_weight = param.sub_view(13588064, 131072);
     DeviceTensor batch_norm_59_mean = param.sub_view(13719136, 256);
     DeviceTensor batch_norm_59_var = param.sub_view(13719392, 256);
     DeviceTensor batch_norm_59_weight = param.sub_view(13719648, 256);
     DeviceTensor batch_norm_59_bias = param.sub_view(13719904, 256);
     DeviceTensor conv_60_weight = param.sub_view(13720160, 1179648);
     DeviceTensor batch_norm_60_mean = param.sub_view(14899808, 512);
     DeviceTensor batch_norm_60_var = param.sub_view(14900320, 512);
     DeviceTensor batch_norm_60_weight = param.sub_view(14900832, 512);
     DeviceTensor batch_norm_60_bias = param.sub_view(14901344, 512);
     DeviceTensor conv_62_weight = param.sub_view(14901856, 4718592);
     DeviceTensor batch_norm_62_mean = param.sub_view(19620448, 1024);
     DeviceTensor batch_norm_62_var = param.sub_view(19621472, 1024);
     DeviceTensor batch_norm_62_weight = param.sub_view(19622496, 1024);
     DeviceTensor batch_norm_62_bias = param.sub_view(19623520, 1024);
     DeviceTensor conv_63_weight = param.sub_view(19624544, 524288);
     DeviceTensor batch_norm_63_mean = param.sub_view(20148832, 512);
     DeviceTensor batch_norm_63_var = param.sub_view(20149344, 512);
     DeviceTensor batch_norm_63_weight = param.sub_view(20149856, 512);
     DeviceTensor batch_norm_63_bias = param.sub_view(20150368, 512);
     DeviceTensor conv_64_weight = param.sub_view(20150880, 4718592);
     DeviceTensor batch_norm_64_mean = param.sub_view(24869472, 1024);
     DeviceTensor batch_norm_64_var = param.sub_view(24870496, 1024);
     DeviceTensor batch_norm_64_weight = param.sub_view(24871520, 1024);
     DeviceTensor batch_norm_64_bias = param.sub_view(24872544, 1024);
     DeviceTensor conv_66_weight = param.sub_view(24873568, 524288);
     DeviceTensor batch_norm_66_mean = param.sub_view(25397856, 512);
     DeviceTensor batch_norm_66_var = param.sub_view(25398368, 512);
     DeviceTensor batch_norm_66_weight = param.sub_view(25398880, 512);
     DeviceTensor batch_norm_66_bias = param.sub_view(25399392, 512);
     DeviceTensor conv_67_weight = param.sub_view(25399904, 4718592);
     DeviceTensor batch_norm_67_mean = param.sub_view(30118496, 1024);
     DeviceTensor batch_norm_67_var = param.sub_view(30119520, 1024);
     DeviceTensor batch_norm_67_weight = param.sub_view(30120544, 1024);
     DeviceTensor batch_norm_67_bias = param.sub_view(30121568, 1024);
     DeviceTensor conv_69_weight = param.sub_view(30122592, 524288);
     DeviceTensor batch_norm_69_mean = param.sub_view(30646880, 512);
     DeviceTensor batch_norm_69_var = param.sub_view(30647392, 512);
     DeviceTensor batch_norm_69_weight = param.sub_view(30647904, 512);
     DeviceTensor batch_norm_69_bias = param.sub_view(30648416, 512);
     DeviceTensor conv_70_weight = param.sub_view(30648928, 4718592);
     DeviceTensor batch_norm_70_mean = param.sub_view(35367520, 1024);
     DeviceTensor batch_norm_70_var = param.sub_view(35368544, 1024);
     DeviceTensor batch_norm_70_weight = param.sub_view(35369568, 1024);
     DeviceTensor batch_norm_70_bias = param.sub_view(35370592, 1024);
     DeviceTensor conv_72_weight = param.sub_view(35371616, 524288);
     DeviceTensor batch_norm_72_mean = param.sub_view(35895904, 512);
     DeviceTensor batch_norm_72_var = param.sub_view(35896416, 512);
     DeviceTensor batch_norm_72_weight = param.sub_view(35896928, 512);
     DeviceTensor batch_norm_72_bias = param.sub_view(35897440, 512);
     DeviceTensor conv_73_weight = param.sub_view(35897952, 4718592);
     DeviceTensor batch_norm_73_mean = param.sub_view(40616544, 1024);
     DeviceTensor batch_norm_73_var = param.sub_view(40617568, 1024);
     DeviceTensor batch_norm_73_weight = param.sub_view(40618592, 1024);
     DeviceTensor batch_norm_73_bias = param.sub_view(40619616, 1024);
     DeviceTensor conv_75_weight = param.sub_view(40620640, 524288);
     DeviceTensor batch_norm_75_mean = param.sub_view(41144928, 512);
     DeviceTensor batch_norm_75_var = param.sub_view(41145440, 512);
     DeviceTensor batch_norm_75_weight = param.sub_view(41145952, 512);
     DeviceTensor batch_norm_75_bias = param.sub_view(41146464, 512);
     DeviceTensor conv_76_weight = param.sub_view(41146976, 4718592);
     DeviceTensor batch_norm_76_mean = param.sub_view(45865568, 1024);
     DeviceTensor batch_norm_76_var = param.sub_view(45866592, 1024);
     DeviceTensor batch_norm_76_weight = param.sub_view(45867616, 1024);
     DeviceTensor batch_norm_76_bias = param.sub_view(45868640, 1024);
     DeviceTensor conv_77_weight = param.sub_view(45869664, 524288);
     DeviceTensor batch_norm_77_mean = param.sub_view(46393952, 512);
     DeviceTensor batch_norm_77_var = param.sub_view(46394464, 512);
     DeviceTensor batch_norm_77_weight = param.sub_view(46394976, 512);
     DeviceTensor batch_norm_77_bias = param.sub_view(46395488, 512);
     DeviceTensor conv_78_weight = param.sub_view(46396000, 4718592);
     DeviceTensor batch_norm_78_mean = param.sub_view(51114592, 1024);
     DeviceTensor batch_norm_78_var = param.sub_view(51115616, 1024);
     DeviceTensor batch_norm_78_weight = param.sub_view(51116640, 1024);
     DeviceTensor batch_norm_78_bias = param.sub_view(51117664, 1024);
     DeviceTensor conv_79_weight = param.sub_view(51118688, 524288);
     DeviceTensor batch_norm_79_mean = param.sub_view(51642976, 512);
     DeviceTensor batch_norm_79_var = param.sub_view(51643488, 512);
     DeviceTensor batch_norm_79_weight = param.sub_view(51644000, 512);
     DeviceTensor batch_norm_79_bias = param.sub_view(51644512, 512);
     DeviceTensor conv_80_weight = param.sub_view(51645024, 4718592);
     DeviceTensor batch_norm_80_mean = param.sub_view(56363616, 1024);
     DeviceTensor batch_norm_80_var = param.sub_view(56364640, 1024);
     DeviceTensor batch_norm_80_weight = param.sub_view(56365664, 1024);
     DeviceTensor batch_norm_80_bias = param.sub_view(56366688, 1024);
     DeviceTensor conv_81_bias = param.sub_view(56367712, 255);
     DeviceTensor conv_81_weight = param.sub_view(56367967, 261120);
     DeviceTensor conv_84_weight = param.sub_view(56629087, 131072);
     DeviceTensor batch_norm_84_mean = param.sub_view(56760159, 256);
     DeviceTensor batch_norm_84_var = param.sub_view(56760415, 256);
     DeviceTensor batch_norm_84_weight = param.sub_view(56760671, 256);
     DeviceTensor batch_norm_84_bias = param.sub_view(56760927, 256);
     DeviceTensor conv_87_weight = param.sub_view(56761183, 196608);
     DeviceTensor batch_norm_87_mean = param.sub_view(56957791, 256);
     DeviceTensor batch_norm_87_var = param.sub_view(56958047, 256);
     DeviceTensor batch_norm_87_weight = param.sub_view(56958303, 256);
     DeviceTensor batch_norm_87_bias = param.sub_view(56958559, 256);
     DeviceTensor conv_88_weight = param.sub_view(56958815, 1179648);
     DeviceTensor batch_norm_88_mean = param.sub_view(58138463, 512);
     DeviceTensor batch_norm_88_var = param.sub_view(58138975, 512);
     DeviceTensor batch_norm_88_weight = param.sub_view(58139487, 512);
     DeviceTensor batch_norm_88_bias = param.sub_view(58139999, 512);
     DeviceTensor conv_89_weight = param.sub_view(58140511, 131072);
     DeviceTensor batch_norm_89_mean = param.sub_view(58271583, 256);
     DeviceTensor batch_norm_89_var = param.sub_view(58271839, 256);
     DeviceTensor batch_norm_89_weight = param.sub_view(58272095, 256);
     DeviceTensor batch_norm_89_bias = param.sub_view(58272351, 256);
     DeviceTensor conv_90_weight = param.sub_view(58272607, 1179648);
     DeviceTensor batch_norm_90_mean = param.sub_view(59452255, 512);
     DeviceTensor batch_norm_90_var = param.sub_view(59452767, 512);
     DeviceTensor batch_norm_90_weight = param.sub_view(59453279, 512);
     DeviceTensor batch_norm_90_bias = param.sub_view(59453791, 512);
     DeviceTensor conv_91_weight = param.sub_view(59454303, 131072);
     DeviceTensor batch_norm_91_mean = param.sub_view(59585375, 256);
     DeviceTensor batch_norm_91_var = param.sub_view(59585631, 256);
     DeviceTensor batch_norm_91_weight = param.sub_view(59585887, 256);
     DeviceTensor batch_norm_91_bias = param.sub_view(59586143, 256);
     DeviceTensor conv_92_weight = param.sub_view(59586399, 1179648);
     DeviceTensor batch_norm_92_mean = param.sub_view(60766047, 512);
     DeviceTensor batch_norm_92_var = param.sub_view(60766559, 512);
     DeviceTensor batch_norm_92_weight = param.sub_view(60767071, 512);
     DeviceTensor batch_norm_92_bias = param.sub_view(60767583, 512);
     DeviceTensor conv_93_bias = param.sub_view(60768095, 255);
     DeviceTensor conv_93_weight = param.sub_view(60768350, 130560);
     DeviceTensor conv_96_weight = param.sub_view(60898910, 32768);
     DeviceTensor batch_norm_96_mean = param.sub_view(60931678, 128);
     DeviceTensor batch_norm_96_var = param.sub_view(60931806, 128);
     DeviceTensor batch_norm_96_weight = param.sub_view(60931934, 128);
     DeviceTensor batch_norm_96_bias = param.sub_view(60932062, 128);
     DeviceTensor conv_99_weight = param.sub_view(60932190, 49152);
     DeviceTensor batch_norm_99_mean = param.sub_view(60981342, 128);
     DeviceTensor batch_norm_99_var = param.sub_view(60981470, 128);
     DeviceTensor batch_norm_99_weight = param.sub_view(60981598, 128);
     DeviceTensor batch_norm_99_bias = param.sub_view(60981726, 128);
     DeviceTensor conv_100_weight = param.sub_view(60981854, 294912);
     DeviceTensor batch_norm_100_mean = param.sub_view(61276766, 256);
     DeviceTensor batch_norm_100_var = param.sub_view(61277022, 256);
     DeviceTensor batch_norm_100_weight = param.sub_view(61277278, 256);
     DeviceTensor batch_norm_100_bias = param.sub_view(61277534, 256);
     DeviceTensor conv_101_weight = param.sub_view(61277790, 32768);
     DeviceTensor batch_norm_101_mean = param.sub_view(61310558, 128);
     DeviceTensor batch_norm_101_var = param.sub_view(61310686, 128);
     DeviceTensor batch_norm_101_weight = param.sub_view(61310814, 128);
     DeviceTensor batch_norm_101_bias = param.sub_view(61310942, 128);
     DeviceTensor conv_102_weight = param.sub_view(61311070, 294912);
     DeviceTensor batch_norm_102_mean = param.sub_view(61605982, 256);
     DeviceTensor batch_norm_102_var = param.sub_view(61606238, 256);
     DeviceTensor batch_norm_102_weight = param.sub_view(61606494, 256);
     DeviceTensor batch_norm_102_bias = param.sub_view(61606750, 256);
     DeviceTensor conv_103_weight = param.sub_view(61607006, 32768);
     DeviceTensor batch_norm_103_mean = param.sub_view(61639774, 128);
     DeviceTensor batch_norm_103_var = param.sub_view(61639902, 128);
     DeviceTensor batch_norm_103_weight = param.sub_view(61640030, 128);
     DeviceTensor batch_norm_103_bias = param.sub_view(61640158, 128);
     DeviceTensor conv_104_weight = param.sub_view(61640286, 294912);
     DeviceTensor batch_norm_104_mean = param.sub_view(61935198, 256);
     DeviceTensor batch_norm_104_var = param.sub_view(61935454, 256);
     DeviceTensor batch_norm_104_weight = param.sub_view(61935710, 256);
     DeviceTensor batch_norm_104_bias = param.sub_view(61935966, 256);
     DeviceTensor conv_105_bias = param.sub_view(61936222, 255);
     DeviceTensor conv_105_weight = param.sub_view(61936477, 65280);
    LibDNNConvOp<DeviceTensor> conv_0 = LibDNNConvOp<DeviceTensor> ("conv_0_forward", 5537792, &conv_0_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {21632,8,1});
    BatchNormOp<DeviceTensor> batch_norm_0 = BatchNormOp<DeviceTensor> (32, 173056, 1e-05, &batch_norm_0_mean, &batch_norm_0_var, &batch_norm_0_weight, &batch_norm_0_bias);
    ReLUOp<DeviceTensor> leaky_0 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_1 = LibDNNConvOp<DeviceTensor> ("conv_1_forward", 2768896, &conv_1_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {5408,16,1});
    BatchNormOp<DeviceTensor> batch_norm_1 = BatchNormOp<DeviceTensor> (64, 43264, 1e-05, &batch_norm_1_mean, &batch_norm_1_var, &batch_norm_1_weight, &batch_norm_1_bias);
    ReLUOp<DeviceTensor> leaky_1 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_2 = LibDNNConvOp<DeviceTensor> ("conv_2_forward", 1384448, &conv_2_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {5408,8,1});
    BatchNormOp<DeviceTensor> batch_norm_2 = BatchNormOp<DeviceTensor> (32, 43264, 1e-05, &batch_norm_2_mean, &batch_norm_2_var, &batch_norm_2_weight, &batch_norm_2_bias);
    ReLUOp<DeviceTensor> leaky_2 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_3 = LibDNNConvOp<DeviceTensor> ("conv_3_forward", 2768896, &conv_3_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {5408,16,1});
    BatchNormOp<DeviceTensor> batch_norm_3 = BatchNormOp<DeviceTensor> (64, 43264, 1e-05, &batch_norm_3_mean, &batch_norm_3_var, &batch_norm_3_weight, &batch_norm_3_bias);
    ReLUOp<DeviceTensor> leaky_3 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_5 = LibDNNConvOp<DeviceTensor> ("conv_5_forward", 1384448, &conv_5_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {1360,32,1});
    BatchNormOp<DeviceTensor> batch_norm_5 = BatchNormOp<DeviceTensor> (128, 10816, 1e-05, &batch_norm_5_mean, &batch_norm_5_var, &batch_norm_5_weight, &batch_norm_5_bias);
    ReLUOp<DeviceTensor> leaky_5 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_6 = LibDNNConvOp<DeviceTensor> ("conv_6_forward", 692224, &conv_6_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {1360,16,1});
    BatchNormOp<DeviceTensor> batch_norm_6 = BatchNormOp<DeviceTensor> (64, 10816, 1e-05, &batch_norm_6_mean, &batch_norm_6_var, &batch_norm_6_weight, &batch_norm_6_bias);
    ReLUOp<DeviceTensor> leaky_6 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_7 = LibDNNConvOp<DeviceTensor> ("conv_7_forward", 1384448, &conv_7_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {1360,32,1});
    BatchNormOp<DeviceTensor> batch_norm_7 = BatchNormOp<DeviceTensor> (128, 10816, 1e-05, &batch_norm_7_mean, &batch_norm_7_var, &batch_norm_7_weight, &batch_norm_7_bias);
    ReLUOp<DeviceTensor> leaky_7 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_9 = LibDNNConvOp<DeviceTensor> ("conv_9_forward", 692224, &conv_9_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {1360,16,1});
    BatchNormOp<DeviceTensor> batch_norm_9 = BatchNormOp<DeviceTensor> (64, 10816, 1e-05, &batch_norm_9_mean, &batch_norm_9_var, &batch_norm_9_weight, &batch_norm_9_bias);
    ReLUOp<DeviceTensor> leaky_9 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_10 = LibDNNConvOp<DeviceTensor> ("conv_10_forward", 1384448, &conv_10_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {1360,32,1});
    BatchNormOp<DeviceTensor> batch_norm_10 = BatchNormOp<DeviceTensor> (128, 10816, 1e-05, &batch_norm_10_mean, &batch_norm_10_var, &batch_norm_10_weight, &batch_norm_10_bias);
    ReLUOp<DeviceTensor> leaky_10 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_12 = LibDNNConvOp<DeviceTensor> ("conv_12_forward", 692224, &conv_12_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> batch_norm_12 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &batch_norm_12_mean, &batch_norm_12_var, &batch_norm_12_weight, &batch_norm_12_bias);
    ReLUOp<DeviceTensor> leaky_12 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_13 = LibDNNConvOp<DeviceTensor> ("conv_13_forward", 346112, &conv_13_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,32,1});
    BatchNormOp<DeviceTensor> batch_norm_13 = BatchNormOp<DeviceTensor> (128, 2704, 1e-05, &batch_norm_13_mean, &batch_norm_13_var, &batch_norm_13_weight, &batch_norm_13_bias);
    ReLUOp<DeviceTensor> leaky_13 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_14 = LibDNNConvOp<DeviceTensor> ("conv_14_forward", 692224, &conv_14_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> batch_norm_14 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &batch_norm_14_mean, &batch_norm_14_var, &batch_norm_14_weight, &batch_norm_14_bias);
    ReLUOp<DeviceTensor> leaky_14 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_16 = LibDNNConvOp<DeviceTensor> ("conv_16_forward", 346112, &conv_16_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,32,1});
    BatchNormOp<DeviceTensor> batch_norm_16 = BatchNormOp<DeviceTensor> (128, 2704, 1e-05, &batch_norm_16_mean, &batch_norm_16_var, &batch_norm_16_weight, &batch_norm_16_bias);
    ReLUOp<DeviceTensor> leaky_16 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_17 = LibDNNConvOp<DeviceTensor> ("conv_17_forward", 692224, &conv_17_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> batch_norm_17 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &batch_norm_17_mean, &batch_norm_17_var, &batch_norm_17_weight, &batch_norm_17_bias);
    ReLUOp<DeviceTensor> leaky_17 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_19 = LibDNNConvOp<DeviceTensor> ("conv_19_forward", 346112, &conv_19_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,32,1});
    BatchNormOp<DeviceTensor> batch_norm_19 = BatchNormOp<DeviceTensor> (128, 2704, 1e-05, &batch_norm_19_mean, &batch_norm_19_var, &batch_norm_19_weight, &batch_norm_19_bias);
    ReLUOp<DeviceTensor> leaky_19 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_20 = LibDNNConvOp<DeviceTensor> ("conv_20_forward", 692224, &conv_20_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> batch_norm_20 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &batch_norm_20_mean, &batch_norm_20_var, &batch_norm_20_weight, &batch_norm_20_bias);
    ReLUOp<DeviceTensor> leaky_20 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_22 = LibDNNConvOp<DeviceTensor> ("conv_22_forward", 346112, &conv_22_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,32,1});
    BatchNormOp<DeviceTensor> batch_norm_22 = BatchNormOp<DeviceTensor> (128, 2704, 1e-05, &batch_norm_22_mean, &batch_norm_22_var, &batch_norm_22_weight, &batch_norm_22_bias);
    ReLUOp<DeviceTensor> leaky_22 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_23 = LibDNNConvOp<DeviceTensor> ("conv_23_forward", 692224, &conv_23_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> batch_norm_23 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &batch_norm_23_mean, &batch_norm_23_var, &batch_norm_23_weight, &batch_norm_23_bias);
    ReLUOp<DeviceTensor> leaky_23 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_25 = LibDNNConvOp<DeviceTensor> ("conv_25_forward", 346112, &conv_25_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,32,1});
    BatchNormOp<DeviceTensor> batch_norm_25 = BatchNormOp<DeviceTensor> (128, 2704, 1e-05, &batch_norm_25_mean, &batch_norm_25_var, &batch_norm_25_weight, &batch_norm_25_bias);
    ReLUOp<DeviceTensor> leaky_25 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_26 = LibDNNConvOp<DeviceTensor> ("conv_26_forward", 692224, &conv_26_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> batch_norm_26 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &batch_norm_26_mean, &batch_norm_26_var, &batch_norm_26_weight, &batch_norm_26_bias);
    ReLUOp<DeviceTensor> leaky_26 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_28 = LibDNNConvOp<DeviceTensor> ("conv_28_forward", 346112, &conv_28_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,32,1});
    BatchNormOp<DeviceTensor> batch_norm_28 = BatchNormOp<DeviceTensor> (128, 2704, 1e-05, &batch_norm_28_mean, &batch_norm_28_var, &batch_norm_28_weight, &batch_norm_28_bias);
    ReLUOp<DeviceTensor> leaky_28 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_29 = LibDNNConvOp<DeviceTensor> ("conv_29_forward", 692224, &conv_29_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> batch_norm_29 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &batch_norm_29_mean, &batch_norm_29_var, &batch_norm_29_weight, &batch_norm_29_bias);
    ReLUOp<DeviceTensor> leaky_29 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_31 = LibDNNConvOp<DeviceTensor> ("conv_31_forward", 346112, &conv_31_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,32,1});
    BatchNormOp<DeviceTensor> batch_norm_31 = BatchNormOp<DeviceTensor> (128, 2704, 1e-05, &batch_norm_31_mean, &batch_norm_31_var, &batch_norm_31_weight, &batch_norm_31_bias);
    ReLUOp<DeviceTensor> leaky_31 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_32 = LibDNNConvOp<DeviceTensor> ("conv_32_forward", 692224, &conv_32_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> batch_norm_32 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &batch_norm_32_mean, &batch_norm_32_var, &batch_norm_32_weight, &batch_norm_32_bias);
    ReLUOp<DeviceTensor> leaky_32 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_34 = LibDNNConvOp<DeviceTensor> ("conv_34_forward", 346112, &conv_34_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,32,1});
    BatchNormOp<DeviceTensor> batch_norm_34 = BatchNormOp<DeviceTensor> (128, 2704, 1e-05, &batch_norm_34_mean, &batch_norm_34_var, &batch_norm_34_weight, &batch_norm_34_bias);
    ReLUOp<DeviceTensor> leaky_34 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_35 = LibDNNConvOp<DeviceTensor> ("conv_35_forward", 692224, &conv_35_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> batch_norm_35 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &batch_norm_35_mean, &batch_norm_35_var, &batch_norm_35_weight, &batch_norm_35_bias);
    ReLUOp<DeviceTensor> leaky_35 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_37 = LibDNNConvOp<DeviceTensor> ("conv_37_forward", 346112, &conv_37_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> batch_norm_37 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &batch_norm_37_mean, &batch_norm_37_var, &batch_norm_37_weight, &batch_norm_37_bias);
    ReLUOp<DeviceTensor> leaky_37 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_38 = LibDNNConvOp<DeviceTensor> ("conv_38_forward", 173056, &conv_38_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    BatchNormOp<DeviceTensor> batch_norm_38 = BatchNormOp<DeviceTensor> (256, 676, 1e-05, &batch_norm_38_mean, &batch_norm_38_var, &batch_norm_38_weight, &batch_norm_38_bias);
    ReLUOp<DeviceTensor> leaky_38 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_39 = LibDNNConvOp<DeviceTensor> ("conv_39_forward", 346112, &conv_39_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> batch_norm_39 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &batch_norm_39_mean, &batch_norm_39_var, &batch_norm_39_weight, &batch_norm_39_bias);
    ReLUOp<DeviceTensor> leaky_39 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_41 = LibDNNConvOp<DeviceTensor> ("conv_41_forward", 173056, &conv_41_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    BatchNormOp<DeviceTensor> batch_norm_41 = BatchNormOp<DeviceTensor> (256, 676, 1e-05, &batch_norm_41_mean, &batch_norm_41_var, &batch_norm_41_weight, &batch_norm_41_bias);
    ReLUOp<DeviceTensor> leaky_41 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_42 = LibDNNConvOp<DeviceTensor> ("conv_42_forward", 346112, &conv_42_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> batch_norm_42 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &batch_norm_42_mean, &batch_norm_42_var, &batch_norm_42_weight, &batch_norm_42_bias);
    ReLUOp<DeviceTensor> leaky_42 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_44 = LibDNNConvOp<DeviceTensor> ("conv_44_forward", 173056, &conv_44_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    BatchNormOp<DeviceTensor> batch_norm_44 = BatchNormOp<DeviceTensor> (256, 676, 1e-05, &batch_norm_44_mean, &batch_norm_44_var, &batch_norm_44_weight, &batch_norm_44_bias);
    ReLUOp<DeviceTensor> leaky_44 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_45 = LibDNNConvOp<DeviceTensor> ("conv_45_forward", 346112, &conv_45_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> batch_norm_45 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &batch_norm_45_mean, &batch_norm_45_var, &batch_norm_45_weight, &batch_norm_45_bias);
    ReLUOp<DeviceTensor> leaky_45 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_47 = LibDNNConvOp<DeviceTensor> ("conv_47_forward", 173056, &conv_47_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    BatchNormOp<DeviceTensor> batch_norm_47 = BatchNormOp<DeviceTensor> (256, 676, 1e-05, &batch_norm_47_mean, &batch_norm_47_var, &batch_norm_47_weight, &batch_norm_47_bias);
    ReLUOp<DeviceTensor> leaky_47 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_48 = LibDNNConvOp<DeviceTensor> ("conv_48_forward", 346112, &conv_48_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> batch_norm_48 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &batch_norm_48_mean, &batch_norm_48_var, &batch_norm_48_weight, &batch_norm_48_bias);
    ReLUOp<DeviceTensor> leaky_48 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_50 = LibDNNConvOp<DeviceTensor> ("conv_50_forward", 173056, &conv_50_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    BatchNormOp<DeviceTensor> batch_norm_50 = BatchNormOp<DeviceTensor> (256, 676, 1e-05, &batch_norm_50_mean, &batch_norm_50_var, &batch_norm_50_weight, &batch_norm_50_bias);
    ReLUOp<DeviceTensor> leaky_50 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_51 = LibDNNConvOp<DeviceTensor> ("conv_51_forward", 346112, &conv_51_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> batch_norm_51 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &batch_norm_51_mean, &batch_norm_51_var, &batch_norm_51_weight, &batch_norm_51_bias);
    ReLUOp<DeviceTensor> leaky_51 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_53 = LibDNNConvOp<DeviceTensor> ("conv_53_forward", 173056, &conv_53_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    BatchNormOp<DeviceTensor> batch_norm_53 = BatchNormOp<DeviceTensor> (256, 676, 1e-05, &batch_norm_53_mean, &batch_norm_53_var, &batch_norm_53_weight, &batch_norm_53_bias);
    ReLUOp<DeviceTensor> leaky_53 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_54 = LibDNNConvOp<DeviceTensor> ("conv_54_forward", 346112, &conv_54_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> batch_norm_54 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &batch_norm_54_mean, &batch_norm_54_var, &batch_norm_54_weight, &batch_norm_54_bias);
    ReLUOp<DeviceTensor> leaky_54 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_56 = LibDNNConvOp<DeviceTensor> ("conv_56_forward", 173056, &conv_56_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    BatchNormOp<DeviceTensor> batch_norm_56 = BatchNormOp<DeviceTensor> (256, 676, 1e-05, &batch_norm_56_mean, &batch_norm_56_var, &batch_norm_56_weight, &batch_norm_56_bias);
    ReLUOp<DeviceTensor> leaky_56 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_57 = LibDNNConvOp<DeviceTensor> ("conv_57_forward", 346112, &conv_57_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> batch_norm_57 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &batch_norm_57_mean, &batch_norm_57_var, &batch_norm_57_weight, &batch_norm_57_bias);
    ReLUOp<DeviceTensor> leaky_57 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_59 = LibDNNConvOp<DeviceTensor> ("conv_59_forward", 173056, &conv_59_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    BatchNormOp<DeviceTensor> batch_norm_59 = BatchNormOp<DeviceTensor> (256, 676, 1e-05, &batch_norm_59_mean, &batch_norm_59_var, &batch_norm_59_weight, &batch_norm_59_bias);
    ReLUOp<DeviceTensor> leaky_59 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_60 = LibDNNConvOp<DeviceTensor> ("conv_60_forward", 346112, &conv_60_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> batch_norm_60 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &batch_norm_60_mean, &batch_norm_60_var, &batch_norm_60_weight, &batch_norm_60_bias);
    ReLUOp<DeviceTensor> leaky_60 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_62 = LibDNNConvOp<DeviceTensor> ("conv_62_forward", 173056, &conv_62_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,256,1});
    BatchNormOp<DeviceTensor> batch_norm_62 = BatchNormOp<DeviceTensor> (1024, 169, 1e-05, &batch_norm_62_mean, &batch_norm_62_var, &batch_norm_62_weight, &batch_norm_62_bias);
    ReLUOp<DeviceTensor> leaky_62 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_63 = LibDNNConvOp<DeviceTensor> ("conv_63_forward", 86528, &conv_63_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,128,1});
    BatchNormOp<DeviceTensor> batch_norm_63 = BatchNormOp<DeviceTensor> (512, 169, 1e-05, &batch_norm_63_mean, &batch_norm_63_var, &batch_norm_63_weight, &batch_norm_63_bias);
    ReLUOp<DeviceTensor> leaky_63 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_64 = LibDNNConvOp<DeviceTensor> ("conv_64_forward", 173056, &conv_64_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,256,1});
    BatchNormOp<DeviceTensor> batch_norm_64 = BatchNormOp<DeviceTensor> (1024, 169, 1e-05, &batch_norm_64_mean, &batch_norm_64_var, &batch_norm_64_weight, &batch_norm_64_bias);
    ReLUOp<DeviceTensor> leaky_64 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_66 = LibDNNConvOp<DeviceTensor> ("conv_66_forward", 86528, &conv_66_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,128,1});
    BatchNormOp<DeviceTensor> batch_norm_66 = BatchNormOp<DeviceTensor> (512, 169, 1e-05, &batch_norm_66_mean, &batch_norm_66_var, &batch_norm_66_weight, &batch_norm_66_bias);
    ReLUOp<DeviceTensor> leaky_66 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_67 = LibDNNConvOp<DeviceTensor> ("conv_67_forward", 173056, &conv_67_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,256,1});
    BatchNormOp<DeviceTensor> batch_norm_67 = BatchNormOp<DeviceTensor> (1024, 169, 1e-05, &batch_norm_67_mean, &batch_norm_67_var, &batch_norm_67_weight, &batch_norm_67_bias);
    ReLUOp<DeviceTensor> leaky_67 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_69 = LibDNNConvOp<DeviceTensor> ("conv_69_forward", 86528, &conv_69_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,128,1});
    BatchNormOp<DeviceTensor> batch_norm_69 = BatchNormOp<DeviceTensor> (512, 169, 1e-05, &batch_norm_69_mean, &batch_norm_69_var, &batch_norm_69_weight, &batch_norm_69_bias);
    ReLUOp<DeviceTensor> leaky_69 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_70 = LibDNNConvOp<DeviceTensor> ("conv_70_forward", 173056, &conv_70_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,256,1});
    BatchNormOp<DeviceTensor> batch_norm_70 = BatchNormOp<DeviceTensor> (1024, 169, 1e-05, &batch_norm_70_mean, &batch_norm_70_var, &batch_norm_70_weight, &batch_norm_70_bias);
    ReLUOp<DeviceTensor> leaky_70 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_72 = LibDNNConvOp<DeviceTensor> ("conv_72_forward", 86528, &conv_72_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,128,1});
    BatchNormOp<DeviceTensor> batch_norm_72 = BatchNormOp<DeviceTensor> (512, 169, 1e-05, &batch_norm_72_mean, &batch_norm_72_var, &batch_norm_72_weight, &batch_norm_72_bias);
    ReLUOp<DeviceTensor> leaky_72 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_73 = LibDNNConvOp<DeviceTensor> ("conv_73_forward", 173056, &conv_73_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,256,1});
    BatchNormOp<DeviceTensor> batch_norm_73 = BatchNormOp<DeviceTensor> (1024, 169, 1e-05, &batch_norm_73_mean, &batch_norm_73_var, &batch_norm_73_weight, &batch_norm_73_bias);
    ReLUOp<DeviceTensor> leaky_73 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_75 = LibDNNConvOp<DeviceTensor> ("conv_75_forward", 86528, &conv_75_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,128,1});
    BatchNormOp<DeviceTensor> batch_norm_75 = BatchNormOp<DeviceTensor> (512, 169, 1e-05, &batch_norm_75_mean, &batch_norm_75_var, &batch_norm_75_weight, &batch_norm_75_bias);
    ReLUOp<DeviceTensor> leaky_75 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_76 = LibDNNConvOp<DeviceTensor> ("conv_76_forward", 173056, &conv_76_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,256,1});
    BatchNormOp<DeviceTensor> batch_norm_76 = BatchNormOp<DeviceTensor> (1024, 169, 1e-05, &batch_norm_76_mean, &batch_norm_76_var, &batch_norm_76_weight, &batch_norm_76_bias);
    ReLUOp<DeviceTensor> leaky_76 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_77 = LibDNNConvOp<DeviceTensor> ("conv_77_forward", 86528, &conv_77_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,128,1});
    BatchNormOp<DeviceTensor> batch_norm_77 = BatchNormOp<DeviceTensor> (512, 169, 1e-05, &batch_norm_77_mean, &batch_norm_77_var, &batch_norm_77_weight, &batch_norm_77_bias);
    ReLUOp<DeviceTensor> leaky_77 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_78 = LibDNNConvOp<DeviceTensor> ("conv_78_forward", 173056, &conv_78_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,256,1});
    BatchNormOp<DeviceTensor> batch_norm_78 = BatchNormOp<DeviceTensor> (1024, 169, 1e-05, &batch_norm_78_mean, &batch_norm_78_var, &batch_norm_78_weight, &batch_norm_78_bias);
    ReLUOp<DeviceTensor> leaky_78 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_79 = LibDNNConvOp<DeviceTensor> ("conv_79_forward", 86528, &conv_79_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,128,1});
    BatchNormOp<DeviceTensor> batch_norm_79 = BatchNormOp<DeviceTensor> (512, 169, 1e-05, &batch_norm_79_mean, &batch_norm_79_var, &batch_norm_79_weight, &batch_norm_79_bias);
    ReLUOp<DeviceTensor> leaky_79 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_80 = LibDNNConvOp<DeviceTensor> ("conv_80_forward", 173056, &conv_80_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,256,1});
    BatchNormOp<DeviceTensor> batch_norm_80 = BatchNormOp<DeviceTensor> (1024, 169, 1e-05, &batch_norm_80_mean, &batch_norm_80_var, &batch_norm_80_weight, &batch_norm_80_bias);
    ReLUOp<DeviceTensor> leaky_80 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_81 = LibDNNConvOp<DeviceTensor> ("conv_81_forward", 43095, &conv_81_weight, &conv_81_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,64,1});
    LibDNNConvOp<DeviceTensor> conv_84 = LibDNNConvOp<DeviceTensor> ("conv_84_forward", 43264, &conv_84_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {32,64,1});
    UpSampling2D<DeviceTensor> upsampling_85 = UpSampling2D<DeviceTensor>(2, 13, 13);
    BatchNormOp<DeviceTensor> batch_norm_84 = BatchNormOp<DeviceTensor> (256, 169, 1e-05, &batch_norm_84_mean, &batch_norm_84_var, &batch_norm_84_weight, &batch_norm_84_bias);
    ReLUOp<DeviceTensor> leaky_84 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_87 = LibDNNConvOp<DeviceTensor> ("conv_87_forward", 173056, &conv_87_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    BatchNormOp<DeviceTensor> batch_norm_87 = BatchNormOp<DeviceTensor> (256, 676, 1e-05, &batch_norm_87_mean, &batch_norm_87_var, &batch_norm_87_weight, &batch_norm_87_bias);
    ReLUOp<DeviceTensor> leaky_87 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_88 = LibDNNConvOp<DeviceTensor> ("conv_88_forward", 346112, &conv_88_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> batch_norm_88 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &batch_norm_88_mean, &batch_norm_88_var, &batch_norm_88_weight, &batch_norm_88_bias);
    ReLUOp<DeviceTensor> leaky_88 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_89 = LibDNNConvOp<DeviceTensor> ("conv_89_forward", 173056, &conv_89_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    BatchNormOp<DeviceTensor> batch_norm_89 = BatchNormOp<DeviceTensor> (256, 676, 1e-05, &batch_norm_89_mean, &batch_norm_89_var, &batch_norm_89_weight, &batch_norm_89_bias);
    ReLUOp<DeviceTensor> leaky_89 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_90 = LibDNNConvOp<DeviceTensor> ("conv_90_forward", 346112, &conv_90_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> batch_norm_90 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &batch_norm_90_mean, &batch_norm_90_var, &batch_norm_90_weight, &batch_norm_90_bias);
    ReLUOp<DeviceTensor> leaky_90 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_91 = LibDNNConvOp<DeviceTensor> ("conv_91_forward", 173056, &conv_91_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    BatchNormOp<DeviceTensor> batch_norm_91 = BatchNormOp<DeviceTensor> (256, 676, 1e-05, &batch_norm_91_mean, &batch_norm_91_var, &batch_norm_91_weight, &batch_norm_91_bias);
    ReLUOp<DeviceTensor> leaky_91 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_92 = LibDNNConvOp<DeviceTensor> ("conv_92_forward", 346112, &conv_92_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,128,1});
    BatchNormOp<DeviceTensor> batch_norm_92 = BatchNormOp<DeviceTensor> (512, 676, 1e-05, &batch_norm_92_mean, &batch_norm_92_var, &batch_norm_92_weight, &batch_norm_92_bias);
    ReLUOp<DeviceTensor> leaky_92 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_93 = LibDNNConvOp<DeviceTensor> ("conv_93_forward", 172380, &conv_93_weight, &conv_93_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,64,1});
    LibDNNConvOp<DeviceTensor> conv_96 = LibDNNConvOp<DeviceTensor> ("conv_96_forward", 86528, &conv_96_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {96,32,1});
    BatchNormOp<DeviceTensor> batch_norm_96 = BatchNormOp<DeviceTensor> (128, 676, 1e-05, &batch_norm_96_mean, &batch_norm_96_var, &batch_norm_96_weight, &batch_norm_96_bias);
    ReLUOp<DeviceTensor> leaky_96 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    UpSampling2D<DeviceTensor> upsampling_97 = UpSampling2D<DeviceTensor>(2, 26, 26);
    LibDNNConvOp<DeviceTensor> conv_99 = LibDNNConvOp<DeviceTensor> ("conv_99_forward", 346112, &conv_99_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,32,1});
    BatchNormOp<DeviceTensor> batch_norm_99 = BatchNormOp<DeviceTensor> (128, 2704, 1e-05, &batch_norm_99_mean, &batch_norm_99_var, &batch_norm_99_weight, &batch_norm_99_bias);
    ReLUOp<DeviceTensor> leaky_99 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_100 = LibDNNConvOp<DeviceTensor> ("conv_100_forward", 692224, &conv_100_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> batch_norm_100 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &batch_norm_100_mean, &batch_norm_100_var, &batch_norm_100_weight, &batch_norm_100_bias);
    ReLUOp<DeviceTensor> leaky_100 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_101 = LibDNNConvOp<DeviceTensor> ("conv_101_forward", 346112, &conv_101_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,32,1});
    BatchNormOp<DeviceTensor> batch_norm_101 = BatchNormOp<DeviceTensor> (128, 2704, 1e-05, &batch_norm_101_mean, &batch_norm_101_var, &batch_norm_101_weight, &batch_norm_101_bias);
    ReLUOp<DeviceTensor> leaky_101 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_102 = LibDNNConvOp<DeviceTensor> ("conv_102_forward", 692224, &conv_102_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> batch_norm_102 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &batch_norm_102_mean, &batch_norm_102_var, &batch_norm_102_weight, &batch_norm_102_bias);
    ReLUOp<DeviceTensor> leaky_102 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_103 = LibDNNConvOp<DeviceTensor> ("conv_103_forward", 346112, &conv_103_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,32,1});
    BatchNormOp<DeviceTensor> batch_norm_103 = BatchNormOp<DeviceTensor> (128, 2704, 1e-05, &batch_norm_103_mean, &batch_norm_103_var, &batch_norm_103_weight, &batch_norm_103_bias);
    ReLUOp<DeviceTensor> leaky_103 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_104 = LibDNNConvOp<DeviceTensor> ("conv_104_forward", 692224, &conv_104_weight, nullptr, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});
    BatchNormOp<DeviceTensor> batch_norm_104 = BatchNormOp<DeviceTensor> (256, 2704, 1e-05, &batch_norm_104_mean, &batch_norm_104_var, &batch_norm_104_weight, &batch_norm_104_bias);
    ReLUOp<DeviceTensor> leaky_104 = ReLUOp<DeviceTensor> ( 0.1, IN_PLACE );
    LibDNNConvOp<DeviceTensor> conv_105 = LibDNNConvOp<DeviceTensor> ("conv_105_forward", 689520, &conv_105_weight, &conv_105_bias, std::vector<size_t> {16,4,1}, std::vector<size_t> {352,64,1});

};


} //namespace hypertea