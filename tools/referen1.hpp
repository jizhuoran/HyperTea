#include "hypertea/hypertea.hpp"

namespace hypertea {

class new_net

{
public:
new_net() { 

FILE *f = fopen("new_net.weight", "rb");
size_t read_size = fread(all_weights, 1, weight_size, f);
if (read_size != weight_size) {  LOG(ERROR) << "Weight File Size Mismatch" << read_size << " and " << weight_size << std::endl;
}
fclose(f);



}


~new_net() {

free(all_weights);

}
void inference( std::vector<float> &data_from_user, std::vector<float> &deconv5_3_to_user) { 


float* conv1_data = (float *)malloc(67108864);
float* conv2_data = (float *)malloc(33554432);
float* conv3_data = (float *)malloc(16777216);
float* conv3_scale3_0_split_0_data = (float *)malloc(16777216);
float* conv3_scale3_0_split_1_data = (float *)malloc(16777216);
float* data_data = (float *)malloc(6291456);
float* deconv5_1_data = (float *)malloc(33554432);
float* deconv5_2_data = (float *)malloc(67108864);
float* deconv5_3_data = (float *)malloc(6291456);
float* res1_conv1_data = (float *)malloc(16777216);
float* res1_conv2_data = (float *)malloc(16777216);
float* res1_output_data = (float *)malloc(16777216);
float* res1_output_res1_elewise_0_split_0_data = (float *)malloc(16777216);
float* res1_output_res1_elewise_0_split_1_data = (float *)malloc(16777216);
float* res2_conv1_data = (float *)malloc(16777216);
float* res2_conv2_data = (float *)malloc(16777216);
float* res2_output_data = (float *)malloc(16777216);
float* res2_output_res2_elewise_0_split_0_data = (float *)malloc(16777216);
float* res2_output_res2_elewise_0_split_1_data = (float *)malloc(16777216);
float* res3_conv1_data = (float *)malloc(16777216);
float* res3_conv2_data = (float *)malloc(16777216);
float* res3_output_data = (float *)malloc(16777216);
float* res3_output_res3_elewise_0_split_0_data = (float *)malloc(16777216);
float* res3_output_res3_elewise_0_split_1_data = (float *)malloc(16777216);
float* res4_conv1_data = (float *)malloc(16777216);
float* res4_conv2_data = (float *)malloc(16777216);
float* res4_output_data = (float *)malloc(16777216);
float* res4_output_res4_elewise_0_split_0_data = (float *)malloc(16777216);
float* res4_output_res4_elewise_0_split_1_data = (float *)malloc(16777216);
float* res5_conv1_data = (float *)malloc(16777216);
float* res5_conv2_data = (float *)malloc(16777216);
float* res5_output_data = (float *)malloc(16777216);


hypertea_copy(data_from_user.size(), data_from_user.data(), data_data);


conv1.Forward({data_data}, {conv1_data});

ELU1.Forward({conv1_data}, {conv1_data});

bn1.Forward({conv1_data}, {conv1_data});

scale1.Forward({conv1_data}, {conv1_data});

conv2.Forward({conv1_data}, {conv2_data});

ELU2.Forward({conv2_data}, {conv2_data});

bn2.Forward({conv2_data}, {conv2_data});

scale2.Forward({conv2_data}, {conv2_data});

conv3.Forward({conv2_data}, {conv3_data});

ELU3.Forward({conv3_data}, {conv3_data});

bn3.Forward({conv3_data}, {conv3_data});

scale3.Forward({conv3_data}, {conv3_data});

conv3_scale3_0_split.Forward({conv3_data}, {conv3_scale3_0_split_0_data , conv3_scale3_0_split_1_data});

res1_conv1.Forward({conv3_scale3_0_split_0_data}, {res1_conv1_data});

res1_bn1.Forward({res1_conv1_data}, {res1_conv1_data});

res1_scale1.Forward({res1_conv1_data}, {res1_conv1_data});

res1_ReLU1.Forward({res1_conv1_data}, {res1_conv1_data});

res1_conv2.Forward({res1_conv1_data}, {res1_conv2_data});

res1_bn2.Forward({res1_conv2_data}, {res1_conv2_data});

res1_scale2.Forward({res1_conv2_data}, {res1_conv2_data});

res1_elewise.Forward({res1_conv2_data , conv3_scale3_0_split_1_data}, {res1_output_data});

res1_output_res1_elewise_0_split.Forward({res1_output_data}, {res1_output_res1_elewise_0_split_0_data , res1_output_res1_elewise_0_split_1_data});

res2_conv1.Forward({res1_output_res1_elewise_0_split_0_data}, {res2_conv1_data});

res2_bn1.Forward({res2_conv1_data}, {res2_conv1_data});

res2_scale1.Forward({res2_conv1_data}, {res2_conv1_data});

res2_ReLU1.Forward({res2_conv1_data}, {res2_conv1_data});

res2_conv2.Forward({res2_conv1_data}, {res2_conv2_data});

res2_bn2.Forward({res2_conv2_data}, {res2_conv2_data});

res2_scale2.Forward({res2_conv2_data}, {res2_conv2_data});

res2_elewise.Forward({res2_conv2_data , res1_output_res1_elewise_0_split_1_data}, {res2_output_data});

res2_output_res2_elewise_0_split.Forward({res2_output_data}, {res2_output_res2_elewise_0_split_0_data , res2_output_res2_elewise_0_split_1_data});

res3_conv1.Forward({res2_output_res2_elewise_0_split_0_data}, {res3_conv1_data});

res3_bn1.Forward({res3_conv1_data}, {res3_conv1_data});

res3_scale1.Forward({res3_conv1_data}, {res3_conv1_data});

res3_ReLU1.Forward({res3_conv1_data}, {res3_conv1_data});

res3_conv2.Forward({res3_conv1_data}, {res3_conv2_data});

res3_bn2.Forward({res3_conv2_data}, {res3_conv2_data});

res3_scale2.Forward({res3_conv2_data}, {res3_conv2_data});

res3_elewise.Forward({res3_conv2_data , res2_output_res2_elewise_0_split_1_data}, {res3_output_data});

res3_output_res3_elewise_0_split.Forward({res3_output_data}, {res3_output_res3_elewise_0_split_0_data , res3_output_res3_elewise_0_split_1_data});

res4_conv1.Forward({res3_output_res3_elewise_0_split_0_data}, {res4_conv1_data});

res4_bn1.Forward({res4_conv1_data}, {res4_conv1_data});

res4_scale1.Forward({res4_conv1_data}, {res4_conv1_data});

res4_ReLU1.Forward({res4_conv1_data}, {res4_conv1_data});

res4_conv2.Forward({res4_conv1_data}, {res4_conv2_data});

res4_bn2.Forward({res4_conv2_data}, {res4_conv2_data});

res4_scale2.Forward({res4_conv2_data}, {res4_conv2_data});

res4_elewise.Forward({res4_conv2_data , res3_output_res3_elewise_0_split_1_data}, {res4_output_data});

res4_output_res4_elewise_0_split.Forward({res4_output_data}, {res4_output_res4_elewise_0_split_0_data , res4_output_res4_elewise_0_split_1_data});

res5_conv1.Forward({res4_output_res4_elewise_0_split_0_data}, {res5_conv1_data});

res5_bn1.Forward({res5_conv1_data}, {res5_conv1_data});

res5_scale1.Forward({res5_conv1_data}, {res5_conv1_data});

res5_ReLU1.Forward({res5_conv1_data}, {res5_conv1_data});

res5_conv2.Forward({res5_conv1_data}, {res5_conv2_data});

res5_bn2.Forward({res5_conv2_data}, {res5_conv2_data});

res5_scale2.Forward({res5_conv2_data}, {res5_conv2_data});

res5_elewise.Forward({res5_conv2_data , res4_output_res4_elewise_0_split_1_data}, {res5_output_data});

deconv5_1.Forward({res5_output_data}, {deconv5_1_data});

deconv5_1_ELU.Forward({deconv5_1_data}, {deconv5_1_data});

deconv5_1_bn.Forward({deconv5_1_data}, {deconv5_1_data});

deconv5_1_bn_sc.Forward({deconv5_1_data}, {deconv5_1_data});

deconv5_2.Forward({deconv5_1_data}, {deconv5_2_data});

deconv5_2_ELU.Forward({deconv5_2_data}, {deconv5_2_data});

deconv5_2_bn.Forward({deconv5_2_data}, {deconv5_2_data});

deconv5_2_bn_sc.Forward({deconv5_2_data}, {deconv5_2_data});

deconv5_3.Forward({deconv5_2_data}, {deconv5_3_data});

tanh.Forward({deconv5_3_data}, {deconv5_3_data});

image_scale1.Forward({deconv5_3_data}, {deconv5_3_data});

image_scale2.Forward({deconv5_3_data}, {deconv5_3_data});



hypertea_copy(deconv5_3_to_user.size(), deconv5_3_data, deconv5_3_to_user.data());


free(conv1_data);
free(conv2_data);
free(conv3_data);
free(conv3_scale3_0_split_0_data);
free(conv3_scale3_0_split_1_data);
free(data_data);
free(deconv5_1_data);
free(deconv5_2_data);
free(deconv5_3_data);
free(res1_conv1_data);
free(res1_conv2_data);
free(res1_output_data);
free(res1_output_res1_elewise_0_split_0_data);
free(res1_output_res1_elewise_0_split_1_data);
free(res2_conv1_data);
free(res2_conv2_data);
free(res2_output_data);
free(res2_output_res2_elewise_0_split_0_data);
free(res2_output_res2_elewise_0_split_1_data);
free(res3_conv1_data);
free(res3_conv2_data);
free(res3_output_data);
free(res3_output_res3_elewise_0_split_0_data);
free(res3_output_res3_elewise_0_split_1_data);
free(res4_conv1_data);
free(res4_conv2_data);
free(res4_output_data);
free(res4_output_res4_elewise_0_split_0_data);
free(res4_output_res4_elewise_0_split_1_data);
free(res5_conv1_data);
free(res5_conv2_data);
free(res5_output_data);


}


private:


int weight_size = 7285296;
unsigned char* all_weights = (unsigned char*) malloc(weight_size);

float* conv1_weight = reinterpret_cast<float*>(all_weights + 0);
float* conv1_bias = reinterpret_cast<float*>(all_weights + 31104);
float* scale1_scale = reinterpret_cast<float*>(all_weights + 31232);
float* scale1_bias = reinterpret_cast<float*>(all_weights + 31360);
float* conv2_weight = reinterpret_cast<float*>(all_weights + 31488);
float* conv2_bias = reinterpret_cast<float*>(all_weights + 162560);
float* scale2_scale = reinterpret_cast<float*>(all_weights + 162816);
float* scale2_bias = reinterpret_cast<float*>(all_weights + 163072);
float* conv3_weight = reinterpret_cast<float*>(all_weights + 163328);
float* conv3_bias = reinterpret_cast<float*>(all_weights + 687616);
float* scale3_scale = reinterpret_cast<float*>(all_weights + 688128);
float* scale3_bias = reinterpret_cast<float*>(all_weights + 688640);
float* res1_conv1_weight = reinterpret_cast<float*>(all_weights + 689152);
float* res1_scale1_scale = reinterpret_cast<float*>(all_weights + 1278976);
float* res1_scale1_bias = reinterpret_cast<float*>(all_weights + 1279488);
float* res1_conv2_weight = reinterpret_cast<float*>(all_weights + 1280000);
float* res1_scale2_scale = reinterpret_cast<float*>(all_weights + 1869824);
float* res1_scale2_bias = reinterpret_cast<float*>(all_weights + 1870336);
float* res2_conv1_weight = reinterpret_cast<float*>(all_weights + 1870848);
float* res2_scale1_scale = reinterpret_cast<float*>(all_weights + 2460672);
float* res2_scale1_bias = reinterpret_cast<float*>(all_weights + 2461184);
float* res2_conv2_weight = reinterpret_cast<float*>(all_weights + 2461696);
float* res2_scale2_scale = reinterpret_cast<float*>(all_weights + 3051520);
float* res2_scale2_bias = reinterpret_cast<float*>(all_weights + 3052032);
float* res3_conv1_weight = reinterpret_cast<float*>(all_weights + 3052544);
float* res3_scale1_scale = reinterpret_cast<float*>(all_weights + 3642368);
float* res3_scale1_bias = reinterpret_cast<float*>(all_weights + 3642880);
float* res3_conv2_weight = reinterpret_cast<float*>(all_weights + 3643392);
float* res3_scale2_scale = reinterpret_cast<float*>(all_weights + 4233216);
float* res3_scale2_bias = reinterpret_cast<float*>(all_weights + 4233728);
float* res4_conv1_weight = reinterpret_cast<float*>(all_weights + 4234240);
float* res4_scale1_scale = reinterpret_cast<float*>(all_weights + 4824064);
float* res4_scale1_bias = reinterpret_cast<float*>(all_weights + 4824576);
float* res4_conv2_weight = reinterpret_cast<float*>(all_weights + 4825088);
float* res4_scale2_scale = reinterpret_cast<float*>(all_weights + 5414912);
float* res4_scale2_bias = reinterpret_cast<float*>(all_weights + 5415424);
float* res5_conv1_weight = reinterpret_cast<float*>(all_weights + 5415936);
float* res5_scale1_scale = reinterpret_cast<float*>(all_weights + 6005760);
float* res5_scale1_bias = reinterpret_cast<float*>(all_weights + 6006272);
float* res5_conv2_weight = reinterpret_cast<float*>(all_weights + 6006784);
float* res5_scale2_scale = reinterpret_cast<float*>(all_weights + 6596608);
float* res5_scale2_bias = reinterpret_cast<float*>(all_weights + 6597120);
float* deconv5_1_weight = reinterpret_cast<float*>(all_weights + 6597632);
float* deconv5_1_bias = reinterpret_cast<float*>(all_weights + 7121920);
float* deconv5_1_bn_sc_scale = reinterpret_cast<float*>(all_weights + 7122176);
float* deconv5_1_bn_sc_bias = reinterpret_cast<float*>(all_weights + 7122432);
float* deconv5_2_weight = reinterpret_cast<float*>(all_weights + 7122688);
float* deconv5_2_bias = reinterpret_cast<float*>(all_weights + 7253760);
float* deconv5_2_bn_sc_scale = reinterpret_cast<float*>(all_weights + 7253888);
float* deconv5_2_bn_sc_bias = reinterpret_cast<float*>(all_weights + 7254016);
float* deconv5_3_weight = reinterpret_cast<float*>(all_weights + 7254144);
float* deconv5_3_bias = reinterpret_cast<float*>(all_weights + 7285248);
float* image_scale1_scale = reinterpret_cast<float*>(all_weights + 7285260);
float* image_scale1_bias = reinterpret_cast<float*>(all_weights + 7285272);
float* image_scale2_scale = reinterpret_cast<float*>(all_weights + 7285284);


ConvolutionOp_CPU<float> conv1 = ConvolutionOp_CPU<float>( conv1_weight, conv1_bias, 786432, 1, 8388608, 2, 3, 1, 7776, 32, 262144, false, false, 32, 3, 262144, 243, 63700992, 8388608, 2, std::vector<int> {9, 9}, std::vector<int> {1, 1}, std::vector<int> {4, 4}, std::vector<int> {1, 1}, std::vector<int> {3, 512, 512}, std::vector<int> {243, 512, 512});
ELUOp_CPU<float> ELU1 = ELUOp_CPU<float>( 16777216, 1);
BatchNormOp_CPU<float> bn1 = BatchNormOp_CPU<float>( 16777216, 2, 32, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> scale1 = ScaleOp_CPU<float>( 16777216, scale1_bias, scale1_scale, 32, 2, 262144);
ConvolutionOp_CPU<float> conv2 = ConvolutionOp_CPU<float>( conv2_weight, conv2_bias, 8388608, 1, 4194304, 2, 32, 1, 32768, 64, 65536, false, false, 64, 32, 65536, 512, 33554432, 4194304, 2, std::vector<int> {4, 4}, std::vector<int> {2, 2}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {32, 512, 512}, std::vector<int> {512, 256, 256});
ELUOp_CPU<float> ELU2 = ELUOp_CPU<float>( 8388608, 1);
BatchNormOp_CPU<float> bn2 = BatchNormOp_CPU<float>( 8388608, 2, 64, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> scale2 = ScaleOp_CPU<float>( 8388608, scale2_bias, scale2_scale, 64, 2, 65536);
ConvolutionOp_CPU<float> conv3 = ConvolutionOp_CPU<float>( conv3_weight, conv3_bias, 4194304, 1, 2097152, 2, 64, 1, 131072, 128, 16384, false, false, 128, 64, 16384, 1024, 16777216, 2097152, 2, std::vector<int> {4, 4}, std::vector<int> {2, 2}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {64, 256, 256}, std::vector<int> {1024, 128, 128});
ELUOp_CPU<float> ELU3 = ELUOp_CPU<float>( 4194304, 1);
BatchNormOp_CPU<float> bn3 = BatchNormOp_CPU<float>( 4194304, 2, 128, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> scale3 = ScaleOp_CPU<float>( 4194304, scale3_bias, scale3_scale, 128, 2, 16384);
SplitOp_CPU<float> conv3_scale3_0_split = SplitOp_CPU<float>( 4194304);
ConvolutionOp_CPU<float> res1_conv1 = ConvolutionOp_CPU<float>( res1_conv1_weight, NULL, 2097152, 1, 2097152, 2, 128, 1, 147456, 128, 16384, false, false, 128, 128, 16384, 1152, 18874368, 2097152, 2, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {128, 128, 128}, std::vector<int> {1152, 128, 128});
BatchNormOp_CPU<float> res1_bn1 = BatchNormOp_CPU<float>( 4194304, 2, 128, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> res1_scale1 = ScaleOp_CPU<float>( 4194304, res1_scale1_bias, res1_scale1_scale, 128, 2, 16384);
ReLUOp_CPU<float> res1_ReLU1 = ReLUOp_CPU<float>( 4194304, 0);
ConvolutionOp_CPU<float> res1_conv2 = ConvolutionOp_CPU<float>( res1_conv2_weight, NULL, 2097152, 1, 2097152, 2, 128, 1, 147456, 128, 16384, false, false, 128, 128, 16384, 1152, 18874368, 2097152, 2, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {128, 128, 128}, std::vector<int> {1152, 128, 128});
BatchNormOp_CPU<float> res1_bn2 = BatchNormOp_CPU<float>( 4194304, 2, 128, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> res1_scale2 = ScaleOp_CPU<float>( 4194304, res1_scale2_bias, res1_scale2_scale, 128, 2, 16384);
EltwiseOp_CPU<float> res1_elewise = EltwiseOp_CPU<float>( 4194304, 2, hypertea::EltwiseParameter_EltwiseOp_SUM, NULL, std::vector<float> { 1, 1 });
SplitOp_CPU<float> res1_output_res1_elewise_0_split = SplitOp_CPU<float>( 4194304);
ConvolutionOp_CPU<float> res2_conv1 = ConvolutionOp_CPU<float>( res2_conv1_weight, NULL, 2097152, 1, 2097152, 2, 128, 1, 147456, 128, 16384, false, false, 128, 128, 16384, 1152, 18874368, 2097152, 2, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {128, 128, 128}, std::vector<int> {1152, 128, 128});
BatchNormOp_CPU<float> res2_bn1 = BatchNormOp_CPU<float>( 4194304, 2, 128, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> res2_scale1 = ScaleOp_CPU<float>( 4194304, res2_scale1_bias, res2_scale1_scale, 128, 2, 16384);
ReLUOp_CPU<float> res2_ReLU1 = ReLUOp_CPU<float>( 4194304, 0);
ConvolutionOp_CPU<float> res2_conv2 = ConvolutionOp_CPU<float>( res2_conv2_weight, NULL, 2097152, 1, 2097152, 2, 128, 1, 147456, 128, 16384, false, false, 128, 128, 16384, 1152, 18874368, 2097152, 2, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {128, 128, 128}, std::vector<int> {1152, 128, 128});
BatchNormOp_CPU<float> res2_bn2 = BatchNormOp_CPU<float>( 4194304, 2, 128, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> res2_scale2 = ScaleOp_CPU<float>( 4194304, res2_scale2_bias, res2_scale2_scale, 128, 2, 16384);
EltwiseOp_CPU<float> res2_elewise = EltwiseOp_CPU<float>( 4194304, 2, hypertea::EltwiseParameter_EltwiseOp_SUM, NULL, std::vector<float> { 1, 1 });
SplitOp_CPU<float> res2_output_res2_elewise_0_split = SplitOp_CPU<float>( 4194304);
ConvolutionOp_CPU<float> res3_conv1 = ConvolutionOp_CPU<float>( res3_conv1_weight, NULL, 2097152, 1, 2097152, 2, 128, 1, 147456, 128, 16384, false, false, 128, 128, 16384, 1152, 18874368, 2097152, 2, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {128, 128, 128}, std::vector<int> {1152, 128, 128});
BatchNormOp_CPU<float> res3_bn1 = BatchNormOp_CPU<float>( 4194304, 2, 128, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> res3_scale1 = ScaleOp_CPU<float>( 4194304, res3_scale1_bias, res3_scale1_scale, 128, 2, 16384);
ReLUOp_CPU<float> res3_ReLU1 = ReLUOp_CPU<float>( 4194304, 0);
ConvolutionOp_CPU<float> res3_conv2 = ConvolutionOp_CPU<float>( res3_conv2_weight, NULL, 2097152, 1, 2097152, 2, 128, 1, 147456, 128, 16384, false, false, 128, 128, 16384, 1152, 18874368, 2097152, 2, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {128, 128, 128}, std::vector<int> {1152, 128, 128});
BatchNormOp_CPU<float> res3_bn2 = BatchNormOp_CPU<float>( 4194304, 2, 128, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> res3_scale2 = ScaleOp_CPU<float>( 4194304, res3_scale2_bias, res3_scale2_scale, 128, 2, 16384);
EltwiseOp_CPU<float> res3_elewise = EltwiseOp_CPU<float>( 4194304, 2, hypertea::EltwiseParameter_EltwiseOp_SUM, NULL, std::vector<float> { 1, 1 });
SplitOp_CPU<float> res3_output_res3_elewise_0_split = SplitOp_CPU<float>( 4194304);
ConvolutionOp_CPU<float> res4_conv1 = ConvolutionOp_CPU<float>( res4_conv1_weight, NULL, 2097152, 1, 2097152, 2, 128, 1, 147456, 128, 16384, false, false, 128, 128, 16384, 1152, 18874368, 2097152, 2, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {128, 128, 128}, std::vector<int> {1152, 128, 128});
BatchNormOp_CPU<float> res4_bn1 = BatchNormOp_CPU<float>( 4194304, 2, 128, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> res4_scale1 = ScaleOp_CPU<float>( 4194304, res4_scale1_bias, res4_scale1_scale, 128, 2, 16384);
ReLUOp_CPU<float> res4_ReLU1 = ReLUOp_CPU<float>( 4194304, 0);
ConvolutionOp_CPU<float> res4_conv2 = ConvolutionOp_CPU<float>( res4_conv2_weight, NULL, 2097152, 1, 2097152, 2, 128, 1, 147456, 128, 16384, false, false, 128, 128, 16384, 1152, 18874368, 2097152, 2, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {128, 128, 128}, std::vector<int> {1152, 128, 128});
BatchNormOp_CPU<float> res4_bn2 = BatchNormOp_CPU<float>( 4194304, 2, 128, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> res4_scale2 = ScaleOp_CPU<float>( 4194304, res4_scale2_bias, res4_scale2_scale, 128, 2, 16384);
EltwiseOp_CPU<float> res4_elewise = EltwiseOp_CPU<float>( 4194304, 2, hypertea::EltwiseParameter_EltwiseOp_SUM, NULL, std::vector<float> { 1, 1 });
SplitOp_CPU<float> res4_output_res4_elewise_0_split = SplitOp_CPU<float>( 4194304);
ConvolutionOp_CPU<float> res5_conv1 = ConvolutionOp_CPU<float>( res5_conv1_weight, NULL, 2097152, 1, 2097152, 2, 128, 1, 147456, 128, 16384, false, false, 128, 128, 16384, 1152, 18874368, 2097152, 2, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {128, 128, 128}, std::vector<int> {1152, 128, 128});
BatchNormOp_CPU<float> res5_bn1 = BatchNormOp_CPU<float>( 4194304, 2, 128, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> res5_scale1 = ScaleOp_CPU<float>( 4194304, res5_scale1_bias, res5_scale1_scale, 128, 2, 16384);
ReLUOp_CPU<float> res5_ReLU1 = ReLUOp_CPU<float>( 4194304, 0);
ConvolutionOp_CPU<float> res5_conv2 = ConvolutionOp_CPU<float>( res5_conv2_weight, NULL, 2097152, 1, 2097152, 2, 128, 1, 147456, 128, 16384, false, false, 128, 128, 16384, 1152, 18874368, 2097152, 2, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {128, 128, 128}, std::vector<int> {1152, 128, 128});
BatchNormOp_CPU<float> res5_bn2 = BatchNormOp_CPU<float>( 4194304, 2, 128, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> res5_scale2 = ScaleOp_CPU<float>( 4194304, res5_scale2_bias, res5_scale2_scale, 128, 2, 16384);
EltwiseOp_CPU<float> res5_elewise = EltwiseOp_CPU<float>( 4194304, 2, hypertea::EltwiseParameter_EltwiseOp_SUM, NULL, std::vector<float> { 1, 1 });
DeconvolutionOp_CPU<float> deconv5_1 = DeconvolutionOp_CPU<float>( deconv5_1_weight, deconv5_1_bias, 2097152, 1, 4194304, 2, 128, 1, 131072, 64, 65536, false, false, 128, 64, 16384, 1024, 16777216, 2097152, 2, std::vector<int> {4, 4}, std::vector<int> {2, 2}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {64, 256, 256}, std::vector<int> {1024, 128, 128});
ELUOp_CPU<float> deconv5_1_ELU = ELUOp_CPU<float>( 8388608, 1);
BatchNormOp_CPU<float> deconv5_1_bn = BatchNormOp_CPU<float>( 8388608, 2, 64, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> deconv5_1_bn_sc = ScaleOp_CPU<float>( 8388608, deconv5_1_bn_sc_bias, deconv5_1_bn_sc_scale, 64, 2, 65536);
DeconvolutionOp_CPU<float> deconv5_2 = DeconvolutionOp_CPU<float>( deconv5_2_weight, deconv5_2_bias, 4194304, 1, 8388608, 2, 64, 1, 32768, 32, 262144, false, false, 64, 32, 65536, 512, 33554432, 4194304, 2, std::vector<int> {4, 4}, std::vector<int> {2, 2}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {32, 512, 512}, std::vector<int> {512, 256, 256});
ELUOp_CPU<float> deconv5_2_ELU = ELUOp_CPU<float>( 16777216, 1);
BatchNormOp_CPU<float> deconv5_2_bn = BatchNormOp_CPU<float>( 16777216, 2, 32, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> deconv5_2_bn_sc = ScaleOp_CPU<float>( 16777216, deconv5_2_bn_sc_bias, deconv5_2_bn_sc_scale, 32, 2, 262144);
DeconvolutionOp_CPU<float> deconv5_3 = DeconvolutionOp_CPU<float>( deconv5_3_weight, deconv5_3_bias, 8388608, 1, 786432, 2, 32, 1, 7776, 3, 262144, false, false, 32, 3, 262144, 243, 63700992, 8388608, 2, std::vector<int> {9, 9}, std::vector<int> {1, 1}, std::vector<int> {4, 4}, std::vector<int> {1, 1}, std::vector<int> {3, 512, 512}, std::vector<int> {243, 512, 512});
TanHOp_CPU<float> tanh = TanHOp_CPU<float>( 1572864);
ScaleOp_CPU<float> image_scale1 = ScaleOp_CPU<float>( 1572864, image_scale1_bias, image_scale1_scale, 3, 2, 262144);
ScaleOp_CPU<float> image_scale2 = ScaleOp_CPU<float>( 1572864, NULL, image_scale2_scale, 3, 2, 262144);


};
} //namespace hypertea


#include "hypertea/hypertea.hpp"

namespace hypertea {

class new_net {
public:

    new_net() {

        FILE *f = fopen("pytorch_weight", "rb");
        size_t read_size = fread(all_weights, 1, weight_size, f);
        if (read_size != weight_size) { 
            LOG(ERROR) << "Weight File Size Mismatch" << read_size << " and " << weight_size << std::endl;
        }
        fclose(f);
    }


    ~new_net() {
        free(all_weights);
    }

    
    
    void inference( std::vector<float> &data_from_user, std::vector<float> &data_to_user) {
        
        TensorCPU<float> data(data_from_user);

        auto temp = bn1(elu1(conv1(data)));
        temp = bn2(elu2(conv2(temp)));
        temp = bn3(elu3(conv3(temp)));


        temp += res1_bn2(res1_conv2(res1_bn1(res1_relu1(res1_bn1(res1_conv1(temp))))));
        temp += res2_bn2(res2_conv2(res2_bn1(res2_relu1(res2_bn1(res2_conv1(temp))))));
        temp += res3_bn2(res3_conv2(res3_bn1(res3_relu1(res3_bn1(res3_conv1(temp))))));
        temp += res4_bn2(res4_conv2(res4_bn1(res4_relu1(res4_bn1(res4_conv1(temp))))));
        temp += res5_bn2(res5_conv2(res5_bn1(res5_relu1(res5_bn1(res5_conv1(temp))))));


        temp = de_bn1(de_elu1(deconv1(temp)));
        temp = de_bn2(de_elu2(deconv2(temp)));
        temp = de_tanh3(deconv3(temp));


        hypertea_copy(data_to_user.size(), temp.data(), data_to_user.data());

    }


private:
    int weight_size = 7308300;
    unsigned char* all_weights = (unsigned char*) malloc(weight_size);
float* conv1_bias = reinterpret_cast<float*>(all_weights + 0);
float* conv1_weight = reinterpret_cast<float*>(all_weights + 128);
float* bn1_bias = reinterpret_cast<float*>(all_weights + 31232);
float* bn1_weight = reinterpret_cast<float*>(all_weights + 31360);
float* bn1_weight = reinterpret_cast<float*>(all_weights + 31488);
float* bn1_bias = reinterpret_cast<float*>(all_weights + 31616);
float* conv2_bias = reinterpret_cast<float*>(all_weights + 31744);
float* conv2_weight = reinterpret_cast<float*>(all_weights + 32000);
float* bn2_bias = reinterpret_cast<float*>(all_weights + 163072);
float* bn2_weight = reinterpret_cast<float*>(all_weights + 163328);
float* bn2_weight = reinterpret_cast<float*>(all_weights + 163584);
float* bn2_bias = reinterpret_cast<float*>(all_weights + 163840);
float* conv3_bias = reinterpret_cast<float*>(all_weights + 164096);
float* conv3_weight = reinterpret_cast<float*>(all_weights + 164608);
float* bn3_bias = reinterpret_cast<float*>(all_weights + 688896);
float* bn3_weight = reinterpret_cast<float*>(all_weights + 689408);
float* bn3_weight = reinterpret_cast<float*>(all_weights + 689920);
float* bn3_bias = reinterpret_cast<float*>(all_weights + 690432);
float* res1_conv1_weight = reinterpret_cast<float*>(all_weights + 690944);
float* res1_bn1_bias = reinterpret_cast<float*>(all_weights + 1280768);
float* res1_bn1_weight = reinterpret_cast<float*>(all_weights + 1281280);
float* res1_bn1_weight = reinterpret_cast<float*>(all_weights + 1281792);
float* res1_bn1_bias = reinterpret_cast<float*>(all_weights + 1282304);
float* res1_bn1_bias = reinterpret_cast<float*>(all_weights + 1282816);
float* res1_bn1_weight = reinterpret_cast<float*>(all_weights + 1283328);
float* res1_bn1_weight = reinterpret_cast<float*>(all_weights + 1283840);
float* res1_bn1_bias = reinterpret_cast<float*>(all_weights + 1284352);
float* res1_conv2_weight = reinterpret_cast<float*>(all_weights + 1284864);
float* res1_bn2_bias = reinterpret_cast<float*>(all_weights + 1874688);
float* res1_bn2_weight = reinterpret_cast<float*>(all_weights + 1875200);
float* res1_bn2_weight = reinterpret_cast<float*>(all_weights + 1875712);
float* res1_bn2_bias = reinterpret_cast<float*>(all_weights + 1876224);
float* res2_conv1_weight = reinterpret_cast<float*>(all_weights + 1876736);
float* res2_bn1_bias = reinterpret_cast<float*>(all_weights + 2466560);
float* res2_bn1_weight = reinterpret_cast<float*>(all_weights + 2467072);
float* res2_bn1_weight = reinterpret_cast<float*>(all_weights + 2467584);
float* res2_bn1_bias = reinterpret_cast<float*>(all_weights + 2468096);
float* res2_bn1_bias = reinterpret_cast<float*>(all_weights + 2468608);
float* res2_bn1_weight = reinterpret_cast<float*>(all_weights + 2469120);
float* res2_bn1_weight = reinterpret_cast<float*>(all_weights + 2469632);
float* res2_bn1_bias = reinterpret_cast<float*>(all_weights + 2470144);
float* res2_conv2_weight = reinterpret_cast<float*>(all_weights + 2470656);
float* res2_bn2_bias = reinterpret_cast<float*>(all_weights + 3060480);
float* res2_bn2_weight = reinterpret_cast<float*>(all_weights + 3060992);
float* res2_bn2_weight = reinterpret_cast<float*>(all_weights + 3061504);
float* res2_bn2_bias = reinterpret_cast<float*>(all_weights + 3062016);
float* res3_conv1_weight = reinterpret_cast<float*>(all_weights + 3062528);
float* res3_bn1_bias = reinterpret_cast<float*>(all_weights + 3652352);
float* res3_bn1_weight = reinterpret_cast<float*>(all_weights + 3652864);
float* res3_bn1_weight = reinterpret_cast<float*>(all_weights + 3653376);
float* res3_bn1_bias = reinterpret_cast<float*>(all_weights + 3653888);
float* res3_bn1_bias = reinterpret_cast<float*>(all_weights + 3654400);
float* res3_bn1_weight = reinterpret_cast<float*>(all_weights + 3654912);
float* res3_bn1_weight = reinterpret_cast<float*>(all_weights + 3655424);
float* res3_bn1_bias = reinterpret_cast<float*>(all_weights + 3655936);
float* res3_conv2_weight = reinterpret_cast<float*>(all_weights + 3656448);
float* res3_bn2_bias = reinterpret_cast<float*>(all_weights + 4246272);
float* res3_bn2_weight = reinterpret_cast<float*>(all_weights + 4246784);
float* res3_bn2_weight = reinterpret_cast<float*>(all_weights + 4247296);
float* res3_bn2_bias = reinterpret_cast<float*>(all_weights + 4247808);
float* res4_conv1_weight = reinterpret_cast<float*>(all_weights + 4248320);
float* res4_bn1_bias = reinterpret_cast<float*>(all_weights + 4838144);
float* res4_bn1_weight = reinterpret_cast<float*>(all_weights + 4838656);
float* res4_bn1_weight = reinterpret_cast<float*>(all_weights + 4839168);
float* res4_bn1_bias = reinterpret_cast<float*>(all_weights + 4839680);
float* res4_bn1_bias = reinterpret_cast<float*>(all_weights + 4840192);
float* res4_bn1_weight = reinterpret_cast<float*>(all_weights + 4840704);
float* res4_bn1_weight = reinterpret_cast<float*>(all_weights + 4841216);
float* res4_bn1_bias = reinterpret_cast<float*>(all_weights + 4841728);
float* res4_conv2_weight = reinterpret_cast<float*>(all_weights + 4842240);
float* res4_bn2_bias = reinterpret_cast<float*>(all_weights + 5432064);
float* res4_bn2_weight = reinterpret_cast<float*>(all_weights + 5432576);
float* res4_bn2_weight = reinterpret_cast<float*>(all_weights + 5433088);
float* res4_bn2_bias = reinterpret_cast<float*>(all_weights + 5433600);
float* res5_conv1_weight = reinterpret_cast<float*>(all_weights + 5434112);
float* res5_bn1_bias = reinterpret_cast<float*>(all_weights + 6023936);
float* res5_bn1_weight = reinterpret_cast<float*>(all_weights + 6024448);
float* res5_bn1_weight = reinterpret_cast<float*>(all_weights + 6024960);
float* res5_bn1_bias = reinterpret_cast<float*>(all_weights + 6025472);
float* res5_bn1_bias = reinterpret_cast<float*>(all_weights + 6025984);
float* res5_bn1_weight = reinterpret_cast<float*>(all_weights + 6026496);
float* res5_bn1_weight = reinterpret_cast<float*>(all_weights + 6027008);
float* res5_bn1_bias = reinterpret_cast<float*>(all_weights + 6027520);
float* res5_conv2_weight = reinterpret_cast<float*>(all_weights + 6028032);
float* res5_bn2_bias = reinterpret_cast<float*>(all_weights + 6617856);
float* res5_bn2_weight = reinterpret_cast<float*>(all_weights + 6618368);
float* res5_bn2_weight = reinterpret_cast<float*>(all_weights + 6618880);
float* res5_bn2_bias = reinterpret_cast<float*>(all_weights + 6619392);
float* deconv1_bias = reinterpret_cast<float*>(all_weights + 6619904);
float* deconv1_weight = reinterpret_cast<float*>(all_weights + 6620160);
float* de_bn1_bias = reinterpret_cast<float*>(all_weights + 7144448);
float* de_bn1_weight = reinterpret_cast<float*>(all_weights + 7144704);
float* de_bn1_weight = reinterpret_cast<float*>(all_weights + 7144960);
float* de_bn1_bias = reinterpret_cast<float*>(all_weights + 7145216);
float* deconv2_bias = reinterpret_cast<float*>(all_weights + 7145472);
float* deconv2_weight = reinterpret_cast<float*>(all_weights + 7145600);
float* de_bn2_bias = reinterpret_cast<float*>(all_weights + 7276672);
float* de_bn2_weight = reinterpret_cast<float*>(all_weights + 7276800);
float* de_bn2_weight = reinterpret_cast<float*>(all_weights + 7276928);
float* de_bn2_bias = reinterpret_cast<float*>(all_weights + 7277056);
float* deconv3_bias = reinterpret_cast<float*>(all_weights + 7277184);
float* deconv3_weight = reinterpret_cast<float*>(all_weights + 7277196);


ConvolutionOp_CPU<float> conv1 = ConvolutionOp_CPU<float> ({ conv1_weight, conv1_bias, 1, false, std::vector<int> {9,9}, std::vector<int> {1,1}, std::vector<int> {4,4}, std::vector<int> {1,1}, std::vector<int> {1,3,512,512}, std::vector<int> {1,32,512,512}, false });
ELUOp_CPU<float> elu1 = ELUOp_CPU<float> ({ 1, NOT_IN_PLACE });
BatchNormOp_CPU<float> bn1 = BatchNormOp_CPU<float> ({ 8388608, bn1_bias, bn1_weight, 1, 32, 1e-05, 1, False, NULL, NULL, bn1_weight, bn1_bias });
ConvolutionOp_CPU<float> conv2 = ConvolutionOp_CPU<float> ({ conv2_weight, conv2_bias, 1, false, std::vector<int> {4,4}, std::vector<int> {2,2}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,32,512,512}, std::vector<int> {1,64,256,256}, false });
ELUOp_CPU<float> elu2 = ELUOp_CPU<float> ({ 1, NOT_IN_PLACE });
BatchNormOp_CPU<float> bn2 = BatchNormOp_CPU<float> ({ 4194304, bn2_bias, bn2_weight, 1, 64, 1e-05, 1, False, NULL, NULL, bn2_weight, bn2_bias });
ConvolutionOp_CPU<float> conv3 = ConvolutionOp_CPU<float> ({ conv3_weight, conv3_bias, 1, false, std::vector<int> {4,4}, std::vector<int> {2,2}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,64,256,256}, std::vector<int> {1,128,128,128}, false });
ELUOp_CPU<float> elu3 = ELUOp_CPU<float> ({ 1, NOT_IN_PLACE });
BatchNormOp_CPU<float> bn3 = BatchNormOp_CPU<float> ({ 2097152, bn3_bias, bn3_weight, 1, 128, 1e-05, 1, False, NULL, NULL, bn3_weight, bn3_bias });
ConvolutionOp_CPU<float> res1_conv1 = ConvolutionOp_CPU<float> ({ res1_conv1_weight, NULL, 1, false, std::vector<int> {3,3}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,128,128,128}, std::vector<int> {1,128,128,128}, false });
BatchNormOp_CPU<float> res1_bn1 = BatchNormOp_CPU<float> ({ 2097152, res1_bn1_bias, res1_bn1_weight, 1, 128, 1e-05, 1, False, NULL, NULL, res1_bn1_weight, res1_bn1_bias });
ReLUOp_CPU<float> res1_relu1 = ReLUOp_CPU<float> ({ 0, NOT_IN_PLACE });
BatchNormOp_CPU<float> res1_bn1 = BatchNormOp_CPU<float> ({ 2097152, res1_bn1_bias, res1_bn1_weight, 1, 128, 1e-05, 1, False, NULL, NULL, res1_bn1_weight, res1_bn1_bias });
ConvolutionOp_CPU<float> res1_conv2 = ConvolutionOp_CPU<float> ({ res1_conv2_weight, NULL, 1, false, std::vector<int> {3,3}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,128,128,128}, std::vector<int> {1,128,128,128}, false });
BatchNormOp_CPU<float> res1_bn2 = BatchNormOp_CPU<float> ({ 2097152, res1_bn2_bias, res1_bn2_weight, 1, 128, 1e-05, 1, False, NULL, NULL, res1_bn2_weight, res1_bn2_bias });
ConvolutionOp_CPU<float> res2_conv1 = ConvolutionOp_CPU<float> ({ res2_conv1_weight, NULL, 1, false, std::vector<int> {3,3}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,128,128,128}, std::vector<int> {1,128,128,128}, false });
BatchNormOp_CPU<float> res2_bn1 = BatchNormOp_CPU<float> ({ 2097152, res2_bn1_bias, res2_bn1_weight, 1, 128, 1e-05, 1, False, NULL, NULL, res2_bn1_weight, res2_bn1_bias });
ReLUOp_CPU<float> res2_relu1 = ReLUOp_CPU<float> ({ 0, NOT_IN_PLACE });
BatchNormOp_CPU<float> res2_bn1 = BatchNormOp_CPU<float> ({ 2097152, res2_bn1_bias, res2_bn1_weight, 1, 128, 1e-05, 1, False, NULL, NULL, res2_bn1_weight, res2_bn1_bias });
ConvolutionOp_CPU<float> res2_conv2 = ConvolutionOp_CPU<float> ({ res2_conv2_weight, NULL, 1, false, std::vector<int> {3,3}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,128,128,128}, std::vector<int> {1,128,128,128}, false });
BatchNormOp_CPU<float> res2_bn2 = BatchNormOp_CPU<float> ({ 2097152, res2_bn2_bias, res2_bn2_weight, 1, 128, 1e-05, 1, False, NULL, NULL, res2_bn2_weight, res2_bn2_bias });
ConvolutionOp_CPU<float> res3_conv1 = ConvolutionOp_CPU<float> ({ res3_conv1_weight, NULL, 1, false, std::vector<int> {3,3}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,128,128,128}, std::vector<int> {1,128,128,128}, false });
BatchNormOp_CPU<float> res3_bn1 = BatchNormOp_CPU<float> ({ 2097152, res3_bn1_bias, res3_bn1_weight, 1, 128, 1e-05, 1, False, NULL, NULL, res3_bn1_weight, res3_bn1_bias });
ReLUOp_CPU<float> res3_relu1 = ReLUOp_CPU<float> ({ 0, NOT_IN_PLACE });
BatchNormOp_CPU<float> res3_bn1 = BatchNormOp_CPU<float> ({ 2097152, res3_bn1_bias, res3_bn1_weight, 1, 128, 1e-05, 1, False, NULL, NULL, res3_bn1_weight, res3_bn1_bias });
ConvolutionOp_CPU<float> res3_conv2 = ConvolutionOp_CPU<float> ({ res3_conv2_weight, NULL, 1, false, std::vector<int> {3,3}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,128,128,128}, std::vector<int> {1,128,128,128}, false });
BatchNormOp_CPU<float> res3_bn2 = BatchNormOp_CPU<float> ({ 2097152, res3_bn2_bias, res3_bn2_weight, 1, 128, 1e-05, 1, False, NULL, NULL, res3_bn2_weight, res3_bn2_bias });
ConvolutionOp_CPU<float> res4_conv1 = ConvolutionOp_CPU<float> ({ res4_conv1_weight, NULL, 1, false, std::vector<int> {3,3}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,128,128,128}, std::vector<int> {1,128,128,128}, false });
BatchNormOp_CPU<float> res4_bn1 = BatchNormOp_CPU<float> ({ 2097152, res4_bn1_bias, res4_bn1_weight, 1, 128, 1e-05, 1, False, NULL, NULL, res4_bn1_weight, res4_bn1_bias });
ReLUOp_CPU<float> res4_relu1 = ReLUOp_CPU<float> ({ 0, NOT_IN_PLACE });
BatchNormOp_CPU<float> res4_bn1 = BatchNormOp_CPU<float> ({ 2097152, res4_bn1_bias, res4_bn1_weight, 1, 128, 1e-05, 1, False, NULL, NULL, res4_bn1_weight, res4_bn1_bias });
ConvolutionOp_CPU<float> res4_conv2 = ConvolutionOp_CPU<float> ({ res4_conv2_weight, NULL, 1, false, std::vector<int> {3,3}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,128,128,128}, std::vector<int> {1,128,128,128}, false });
BatchNormOp_CPU<float> res4_bn2 = BatchNormOp_CPU<float> ({ 2097152, res4_bn2_bias, res4_bn2_weight, 1, 128, 1e-05, 1, False, NULL, NULL, res4_bn2_weight, res4_bn2_bias });
ConvolutionOp_CPU<float> res5_conv1 = ConvolutionOp_CPU<float> ({ res5_conv1_weight, NULL, 1, false, std::vector<int> {3,3}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,128,128,128}, std::vector<int> {1,128,128,128}, false });
BatchNormOp_CPU<float> res5_bn1 = BatchNormOp_CPU<float> ({ 2097152, res5_bn1_bias, res5_bn1_weight, 1, 128, 1e-05, 1, False, NULL, NULL, res5_bn1_weight, res5_bn1_bias });
ReLUOp_CPU<float> res5_relu1 = ReLUOp_CPU<float> ({ 0, NOT_IN_PLACE });
BatchNormOp_CPU<float> res5_bn1 = BatchNormOp_CPU<float> ({ 2097152, res5_bn1_bias, res5_bn1_weight, 1, 128, 1e-05, 1, False, NULL, NULL, res5_bn1_weight, res5_bn1_bias });
ConvolutionOp_CPU<float> res5_conv2 = ConvolutionOp_CPU<float> ({ res5_conv2_weight, NULL, 1, false, std::vector<int> {3,3}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,128,128,128}, std::vector<int> {1,128,128,128}, false });
BatchNormOp_CPU<float> res5_bn2 = BatchNormOp_CPU<float> ({ 2097152, res5_bn2_bias, res5_bn2_weight, 1, 128, 1e-05, 1, False, NULL, NULL, res5_bn2_weight, res5_bn2_bias });
DeconvolutionOp_CPU<float> deconv1 = DeconvolutionOp_CPU<float> ({ deconv1_weight, deconv1_bias, 1, false, std::vector<int> {4,4}, std::vector<int> {2,2}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,128,128,128}, std::vector<int> {1,64,256,256}, false });
ELUOp_CPU<float> de_elu1 = ELUOp_CPU<float> ({ 1, NOT_IN_PLACE });
BatchNormOp_CPU<float> de_bn1 = BatchNormOp_CPU<float> ({ 4194304, de_bn1_bias, de_bn1_weight, 1, 64, 1e-05, 1, False, NULL, NULL, de_bn1_weight, de_bn1_bias });
DeconvolutionOp_CPU<float> deconv2 = DeconvolutionOp_CPU<float> ({ deconv2_weight, deconv2_bias, 1, false, std::vector<int> {4,4}, std::vector<int> {2,2}, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {1,64,256,256}, std::vector<int> {1,32,512,512}, false });
ELUOp_CPU<float> de_elu2 = ELUOp_CPU<float> ({ 1, NOT_IN_PLACE });
BatchNormOp_CPU<float> de_bn2 = BatchNormOp_CPU<float> ({ 8388608, de_bn2_bias, de_bn2_weight, 1, 32, 1e-05, 1, False, NULL, NULL, de_bn2_weight, de_bn2_bias });
DeconvolutionOp_CPU<float> deconv3 = DeconvolutionOp_CPU<float> ({ deconv3_weight, deconv3_bias, 1, false, std::vector<int> {9,9}, std::vector<int> {1,1}, std::vector<int> {4,4}, std::vector<int> {1,1}, std::vector<int> {1,32,512,512}, std::vector<int> {1,3,512,512}, false });
TanHOp_CPU<float> de_tanh3 = TanHOp_CPU<float> ({{ NOT_IN_PLACE }});

};
} //namespace hypertea

