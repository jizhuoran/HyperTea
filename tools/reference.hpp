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


Tensor<float> conv1_data(16777216);
Tensor<float> conv2_data(8388608);
Tensor<float> conv3_data(4194304);
Tensor<float> conv3_scale3_0_split_0_data(4194304);
Tensor<float> conv3_scale3_0_split_1_data(4194304);
Tensor<float> data_data(1572864);
Tensor<float> deconv5_1_data(8388608);
Tensor<float> deconv5_2_data(16777216);
Tensor<float> deconv5_3_data(1572864);
Tensor<float> res1_conv1_data(4194304);
Tensor<float> res1_conv2_data(4194304);
Tensor<float> res1_output_data(4194304);
Tensor<float> res1_output_res1_elewise_0_split_0_data(4194304);
Tensor<float> res1_output_res1_elewise_0_split_1_data(4194304);
Tensor<float> res2_conv1_data(4194304);
Tensor<float> res2_conv2_data(4194304);
Tensor<float> res2_output_data(4194304);
Tensor<float> res2_output_res2_elewise_0_split_0_data(4194304);
Tensor<float> res2_output_res2_elewise_0_split_1_data(4194304);
Tensor<float> res3_conv1_data(4194304);
Tensor<float> res3_conv2_data(4194304);
Tensor<float> res3_output_data(4194304);
Tensor<float> res3_output_res3_elewise_0_split_0_data(4194304);
Tensor<float> res3_output_res3_elewise_0_split_1_data(4194304);
Tensor<float> res4_conv1_data(4194304);
Tensor<float> res4_conv2_data(4194304);
Tensor<float> res4_output_data(4194304);
Tensor<float> res4_output_res4_elewise_0_split_0_data(4194304);
Tensor<float> res4_output_res4_elewise_0_split_1_data(4194304);
Tensor<float> res5_conv1_data(4194304);
Tensor<float> res5_conv2_data(4194304);
Tensor<float> res5_output_data(4194304);


hypertea_copy(data_from_user.size(), data_from_user.data(), data_data.data());


conv1_data = conv1(data_data);

conv1_data = ELU1(conv1_data);

conv1_data = bn1(conv1_data);

conv1_data = scale1(conv1_data);

conv2_data = conv2(conv1_data);

conv2_data = ELU2(conv2_data);

conv2_data = bn2(conv2_data);

conv2_data = scale2(conv2_data);

conv3_data = conv3(conv2_data);

conv3_data = ELU3(conv3_data);

conv3_data = bn3(conv3_data);

conv3_data = scale3(conv3_data);

// conv3_scale3_0_split_0_data = conv3_scale3_0_split(conv3_data);

res1_conv1_data = res1_conv1(conv3_data);

res1_conv1_data = res1_bn1(res1_conv1_data);

res1_conv1_data = res1_scale1(res1_conv1_data);

res1_conv1_data = res1_ReLU1(res1_conv1_data);

res1_conv2_data = res1_conv2(res1_conv1_data);

res1_conv2_data = res1_bn2(res1_conv2_data);

res1_conv2_data = res1_scale2(res1_conv2_data);


res1_conv2_data += conv3_data;
// res1_output_data = res1_elewise(res1_conv2_data);

// res1_output_res1_elewise_0_split_0_data = res1_output_res1_elewise_0_split(res1_output_data);

res2_conv1_data = res2_conv1(res1_conv2_data);

res2_conv1_data = res2_bn1(res2_conv1_data);

res2_conv1_data = res2_scale1(res2_conv1_data);

res2_conv1_data = res2_ReLU1(res2_conv1_data);

res2_conv2_data = res2_conv2(res2_conv1_data);

res2_conv2_data = res2_bn2(res2_conv2_data);

res2_conv2_data = res2_scale2(res2_conv2_data);


res2_conv2_data += res1_conv2_data;
// res2_output_data = res2_elewise(res2_conv2_data);

// res2_output_res2_elewise_0_split_0_data = res2_output_res2_elewise_0_split(res2_output_data);

res3_conv1_data = res3_conv1(res2_conv2_data);

res3_conv1_data = res3_bn1(res3_conv1_data);

res3_conv1_data = res3_scale1(res3_conv1_data);

res3_conv1_data = res3_ReLU1(res3_conv1_data);

res3_conv2_data = res3_conv2(res3_conv1_data);

res3_conv2_data = res3_bn2(res3_conv2_data);

res3_conv2_data = res3_scale2(res3_conv2_data);

// res3_output_data = res3_elewise(res3_conv2_data);

// res3_output_res3_elewise_0_split_0_data = res3_output_res3_elewise_0_split(res3_output_data);

res3_conv2_data += res2_conv2_data;


res4_conv1_data = res4_conv1(res3_conv2_data);

res4_conv1_data = res4_bn1(res4_conv1_data);

res4_conv1_data = res4_scale1(res4_conv1_data);

res4_conv1_data = res4_ReLU1(res4_conv1_data);

res4_conv2_data = res4_conv2(res4_conv1_data);

res4_conv2_data = res4_bn2(res4_conv2_data);

res4_conv2_data = res4_scale2(res4_conv2_data);

res4_conv2_data += res3_conv2_data;


// res4_output_data = res4_elewise(res4_conv2_data);

// res4_output_res4_elewise_0_split_0_data = res4_output_res4_elewise_0_split(res4_output_data);

res5_conv1_data = res5_conv1(res4_conv2_data);

res5_conv1_data = res5_bn1(res5_conv1_data);

res5_conv1_data = res5_scale1(res5_conv1_data);

res5_conv1_data = res5_ReLU1(res5_conv1_data);

res5_conv2_data = res5_conv2(res5_conv1_data);

res5_conv2_data = res5_bn2(res5_conv2_data);

res5_conv2_data = res5_scale2(res5_conv2_data);


res5_conv2_data += res4_conv2_data;


// res5_output_data = res5_elewise(res5_conv2_data);

deconv5_1_data = deconv5_1(res5_conv2_data);

deconv5_1_data = deconv5_1_ELU(deconv5_1_data);

deconv5_1_data = deconv5_1_bn(deconv5_1_data);

deconv5_1_data = deconv5_1_bn_sc(deconv5_1_data);

deconv5_2_data = deconv5_2(deconv5_1_data);

deconv5_2_data = deconv5_2_ELU(deconv5_2_data);

deconv5_2_data = deconv5_2_bn(deconv5_2_data);

deconv5_2_data = deconv5_2_bn_sc(deconv5_2_data);

deconv5_3_data = deconv5_3(deconv5_2_data);

deconv5_3_data = tanh(deconv5_3_data);

deconv5_3_data = image_scale1(deconv5_3_data);

deconv5_3_data = image_scale2(deconv5_3_data);



hypertea_copy(deconv5_3_to_user.size(), deconv5_3_data.data(), deconv5_3_to_user.data());


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


ConvolutionOp_CPU<float> conv1 = ConvolutionOp_CPU<float>( conv1_weight, conv1_bias, 1, false, std::vector<int> {9, 9}, std::vector<int> {1, 1}, std::vector<int> {4, 4}, std::vector<int> {1, 1}, std::vector<int> {2, 3, 512, 512}, std::vector<int> {2, 32, 512, 512}, false);
ELUOp_CPU<float> ELU1 = ELUOp_CPU<float>( 16777216, 1);
BatchNormOp_CPU<float> bn1 = BatchNormOp_CPU<float>( 16777216, 2, 32, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> scale1 = ScaleOp_CPU<float>( 16777216, scale1_bias, scale1_scale, 32, 2, 262144);
ConvolutionOp_CPU<float> conv2 = ConvolutionOp_CPU<float>( conv2_weight, conv2_bias, 1, false, std::vector<int> {4, 4}, std::vector<int> {2, 2}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 32, 512, 512}, std::vector<int> {2, 64, 256, 256}, false);
ELUOp_CPU<float> ELU2 = ELUOp_CPU<float>( 8388608, 1);
BatchNormOp_CPU<float> bn2 = BatchNormOp_CPU<float>( 8388608, 2, 64, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> scale2 = ScaleOp_CPU<float>( 8388608, scale2_bias, scale2_scale, 64, 2, 65536);
ConvolutionOp_CPU<float> conv3 = ConvolutionOp_CPU<float>( conv3_weight, conv3_bias, 1, false, std::vector<int> {4, 4}, std::vector<int> {2, 2}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 64, 256, 256}, std::vector<int> {2, 128, 128, 128}, false);
ELUOp_CPU<float> ELU3 = ELUOp_CPU<float>( 4194304, 1);
BatchNormOp_CPU<float> bn3 = BatchNormOp_CPU<float>( 4194304, 2, 128, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> scale3 = ScaleOp_CPU<float>( 4194304, scale3_bias, scale3_scale, 128, 2, 16384);
SplitOp_CPU<float> conv3_scale3_0_split = SplitOp_CPU<float>( 4194304);
ConvolutionOp_CPU<float> res1_conv1 = ConvolutionOp_CPU<float>( res1_conv1_weight, NULL, 1, false, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 128, 128, 128}, std::vector<int> {2, 128, 128, 128}, false);
BatchNormOp_CPU<float> res1_bn1 = BatchNormOp_CPU<float>( 4194304, 2, 128, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> res1_scale1 = ScaleOp_CPU<float>( 4194304, res1_scale1_bias, res1_scale1_scale, 128, 2, 16384);
ReLUOp_CPU<float> res1_ReLU1 = ReLUOp_CPU<float>( 4194304, 0);
ConvolutionOp_CPU<float> res1_conv2 = ConvolutionOp_CPU<float>( res1_conv2_weight, NULL, 1, false, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 128, 128, 128}, std::vector<int> {2, 128, 128, 128}, false);
BatchNormOp_CPU<float> res1_bn2 = BatchNormOp_CPU<float>( 4194304, 2, 128, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> res1_scale2 = ScaleOp_CPU<float>( 4194304, res1_scale2_bias, res1_scale2_scale, 128, 2, 16384);
EltwiseOp_CPU<float> res1_elewise = EltwiseOp_CPU<float>( 4194304, 2, hypertea::EltwiseParameter_EltwiseOp_SUM, NULL, std::vector<float> { 1, 1 });
SplitOp_CPU<float> res1_output_res1_elewise_0_split = SplitOp_CPU<float>( 4194304);
ConvolutionOp_CPU<float> res2_conv1 = ConvolutionOp_CPU<float>( res2_conv1_weight, NULL, 1, false, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 128, 128, 128}, std::vector<int> {2, 128, 128, 128}, false);
BatchNormOp_CPU<float> res2_bn1 = BatchNormOp_CPU<float>( 4194304, 2, 128, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> res2_scale1 = ScaleOp_CPU<float>( 4194304, res2_scale1_bias, res2_scale1_scale, 128, 2, 16384);
ReLUOp_CPU<float> res2_ReLU1 = ReLUOp_CPU<float>( 4194304, 0);
ConvolutionOp_CPU<float> res2_conv2 = ConvolutionOp_CPU<float>( res2_conv2_weight, NULL, 1, false, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 128, 128, 128}, std::vector<int> {2, 128, 128, 128}, false);
BatchNormOp_CPU<float> res2_bn2 = BatchNormOp_CPU<float>( 4194304, 2, 128, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> res2_scale2 = ScaleOp_CPU<float>( 4194304, res2_scale2_bias, res2_scale2_scale, 128, 2, 16384);
EltwiseOp_CPU<float> res2_elewise = EltwiseOp_CPU<float>( 4194304, 2, hypertea::EltwiseParameter_EltwiseOp_SUM, NULL, std::vector<float> { 1, 1 });
SplitOp_CPU<float> res2_output_res2_elewise_0_split = SplitOp_CPU<float>( 4194304);
ConvolutionOp_CPU<float> res3_conv1 = ConvolutionOp_CPU<float>( res3_conv1_weight, NULL, 1, false, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 128, 128, 128}, std::vector<int> {2, 128, 128, 128}, false);
BatchNormOp_CPU<float> res3_bn1 = BatchNormOp_CPU<float>( 4194304, 2, 128, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> res3_scale1 = ScaleOp_CPU<float>( 4194304, res3_scale1_bias, res3_scale1_scale, 128, 2, 16384);
ReLUOp_CPU<float> res3_ReLU1 = ReLUOp_CPU<float>( 4194304, 0);
ConvolutionOp_CPU<float> res3_conv2 = ConvolutionOp_CPU<float>( res3_conv2_weight, NULL, 1, false, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 128, 128, 128}, std::vector<int> {2, 128, 128, 128}, false);
BatchNormOp_CPU<float> res3_bn2 = BatchNormOp_CPU<float>( 4194304, 2, 128, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> res3_scale2 = ScaleOp_CPU<float>( 4194304, res3_scale2_bias, res3_scale2_scale, 128, 2, 16384);
EltwiseOp_CPU<float> res3_elewise = EltwiseOp_CPU<float>( 4194304, 2, hypertea::EltwiseParameter_EltwiseOp_SUM, NULL, std::vector<float> { 1, 1 });
SplitOp_CPU<float> res3_output_res3_elewise_0_split = SplitOp_CPU<float>( 4194304);
ConvolutionOp_CPU<float> res4_conv1 = ConvolutionOp_CPU<float>( res4_conv1_weight, NULL, 1, false, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 128, 128, 128}, std::vector<int> {2, 128, 128, 128}, false);
BatchNormOp_CPU<float> res4_bn1 = BatchNormOp_CPU<float>( 4194304, 2, 128, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> res4_scale1 = ScaleOp_CPU<float>( 4194304, res4_scale1_bias, res4_scale1_scale, 128, 2, 16384);
ReLUOp_CPU<float> res4_ReLU1 = ReLUOp_CPU<float>( 4194304, 0);
ConvolutionOp_CPU<float> res4_conv2 = ConvolutionOp_CPU<float>( res4_conv2_weight, NULL, 1, false, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 128, 128, 128}, std::vector<int> {2, 128, 128, 128}, false);
BatchNormOp_CPU<float> res4_bn2 = BatchNormOp_CPU<float>( 4194304, 2, 128, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> res4_scale2 = ScaleOp_CPU<float>( 4194304, res4_scale2_bias, res4_scale2_scale, 128, 2, 16384);
EltwiseOp_CPU<float> res4_elewise = EltwiseOp_CPU<float>( 4194304, 2, hypertea::EltwiseParameter_EltwiseOp_SUM, NULL, std::vector<float> { 1, 1 });
SplitOp_CPU<float> res4_output_res4_elewise_0_split = SplitOp_CPU<float>( 4194304);
ConvolutionOp_CPU<float> res5_conv1 = ConvolutionOp_CPU<float>( res5_conv1_weight, NULL, 1, false, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 128, 128, 128}, std::vector<int> {2, 128, 128, 128}, false);
BatchNormOp_CPU<float> res5_bn1 = BatchNormOp_CPU<float>( 4194304, 2, 128, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> res5_scale1 = ScaleOp_CPU<float>( 4194304, res5_scale1_bias, res5_scale1_scale, 128, 2, 16384);
ReLUOp_CPU<float> res5_ReLU1 = ReLUOp_CPU<float>( 4194304, 0);
ConvolutionOp_CPU<float> res5_conv2 = ConvolutionOp_CPU<float>( res5_conv2_weight, NULL, 1, false, std::vector<int> {3, 3}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 128, 128, 128}, std::vector<int> {2, 128, 128, 128}, false);
BatchNormOp_CPU<float> res5_bn2 = BatchNormOp_CPU<float>( 4194304, 2, 128, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> res5_scale2 = ScaleOp_CPU<float>( 4194304, res5_scale2_bias, res5_scale2_scale, 128, 2, 16384);
EltwiseOp_CPU<float> res5_elewise = EltwiseOp_CPU<float>( 4194304, 2, hypertea::EltwiseParameter_EltwiseOp_SUM, NULL, std::vector<float> { 1, 1 });
DeconvolutionOp_CPU<float> deconv5_1 = DeconvolutionOp_CPU<float>( deconv5_1_weight, deconv5_1_bias, 1, false, std::vector<int> {4, 4}, std::vector<int> {2, 2}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 128, 128, 128}, std::vector<int> {2, 64, 256, 256}, false);
ELUOp_CPU<float> deconv5_1_ELU = ELUOp_CPU<float>( 8388608, 1);
BatchNormOp_CPU<float> deconv5_1_bn = BatchNormOp_CPU<float>( 8388608, 2, 64, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> deconv5_1_bn_sc = ScaleOp_CPU<float>( 8388608, deconv5_1_bn_sc_bias, deconv5_1_bn_sc_scale, 64, 2, 65536);
DeconvolutionOp_CPU<float> deconv5_2 = DeconvolutionOp_CPU<float>( deconv5_2_weight, deconv5_2_bias, 1, false, std::vector<int> {4, 4}, std::vector<int> {2, 2}, std::vector<int> {1, 1}, std::vector<int> {1, 1}, std::vector<int> {2, 64, 256, 256}, std::vector<int> {2, 32, 512, 512}, false);
ELUOp_CPU<float> deconv5_2_ELU = ELUOp_CPU<float>( 16777216, 1);
BatchNormOp_CPU<float> deconv5_2_bn = BatchNormOp_CPU<float>( 16777216, 2, 32, 1e-05, 0.50025, false, NULL, NULL);
ScaleOp_CPU<float> deconv5_2_bn_sc = ScaleOp_CPU<float>( 16777216, deconv5_2_bn_sc_bias, deconv5_2_bn_sc_scale, 32, 2, 262144);
DeconvolutionOp_CPU<float> deconv5_3 = DeconvolutionOp_CPU<float>( deconv5_3_weight, deconv5_3_bias, 1, false, std::vector<int> {9, 9}, std::vector<int> {1, 1}, std::vector<int> {4, 4}, std::vector<int> {1, 1}, std::vector<int> {2, 32, 512, 512}, std::vector<int> {2, 3, 512, 512}, false);
TanHOp_CPU<float> tanh = TanHOp_CPU<float>( 1572864);
ScaleOp_CPU<float> image_scale1 = ScaleOp_CPU<float>( 1572864, image_scale1_bias, image_scale1_scale, 3, 2, 262144);
ScaleOp_CPU<float> image_scale2 = ScaleOp_CPU<float>( 1572864, NULL, image_scale2_scale, 3, 2, 262144);


};
} //namespace hypertea

