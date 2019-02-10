#include "hypertea/hypertea.hpp"

namespace hypertea {

class new_net

{
public:
new_net() { 

int weight_size = 7285296;
unsigned char* all_weights = (unsigned char*) malloc(weight_size);

FILE *f = fopen("./examples/style_transfer/new_net.weight", "rb");
size_t read_size = fread(all_weights, 1, weight_size, f);
if (read_size != weight_size) {  LOG(ERROR) << "Weight File Size Mismatch" << read_size << " and " << weight_size << std::endl;
}
fclose(f);

OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, conv1_weight, CL_TRUE, 0, 31104, all_weights + 0, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, conv1_bias, CL_TRUE, 0, 128, all_weights + 31104, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, scale1_scale, CL_TRUE, 0, 128, all_weights + 31232, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, scale1_bias, CL_TRUE, 0, 128, all_weights + 31360, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, conv2_weight, CL_TRUE, 0, 131072, all_weights + 31488, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, conv2_bias, CL_TRUE, 0, 256, all_weights + 162560, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, scale2_scale, CL_TRUE, 0, 256, all_weights + 162816, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, scale2_bias, CL_TRUE, 0, 256, all_weights + 163072, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, conv3_weight, CL_TRUE, 0, 524288, all_weights + 163328, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, conv3_bias, CL_TRUE, 0, 512, all_weights + 687616, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, scale3_scale, CL_TRUE, 0, 512, all_weights + 688128, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, scale3_bias, CL_TRUE, 0, 512, all_weights + 688640, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res1_conv1_weight, CL_TRUE, 0, 589824, all_weights + 689152, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res1_scale1_scale, CL_TRUE, 0, 512, all_weights + 1278976, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res1_scale1_bias, CL_TRUE, 0, 512, all_weights + 1279488, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res1_conv2_weight, CL_TRUE, 0, 589824, all_weights + 1280000, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res1_scale2_scale, CL_TRUE, 0, 512, all_weights + 1869824, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res1_scale2_bias, CL_TRUE, 0, 512, all_weights + 1870336, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res2_conv1_weight, CL_TRUE, 0, 589824, all_weights + 1870848, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res2_scale1_scale, CL_TRUE, 0, 512, all_weights + 2460672, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res2_scale1_bias, CL_TRUE, 0, 512, all_weights + 2461184, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res2_conv2_weight, CL_TRUE, 0, 589824, all_weights + 2461696, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res2_scale2_scale, CL_TRUE, 0, 512, all_weights + 3051520, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res2_scale2_bias, CL_TRUE, 0, 512, all_weights + 3052032, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res3_conv1_weight, CL_TRUE, 0, 589824, all_weights + 3052544, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res3_scale1_scale, CL_TRUE, 0, 512, all_weights + 3642368, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res3_scale1_bias, CL_TRUE, 0, 512, all_weights + 3642880, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res3_conv2_weight, CL_TRUE, 0, 589824, all_weights + 3643392, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res3_scale2_scale, CL_TRUE, 0, 512, all_weights + 4233216, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res3_scale2_bias, CL_TRUE, 0, 512, all_weights + 4233728, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res4_conv1_weight, CL_TRUE, 0, 589824, all_weights + 4234240, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res4_scale1_scale, CL_TRUE, 0, 512, all_weights + 4824064, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res4_scale1_bias, CL_TRUE, 0, 512, all_weights + 4824576, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res4_conv2_weight, CL_TRUE, 0, 589824, all_weights + 4825088, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res4_scale2_scale, CL_TRUE, 0, 512, all_weights + 5414912, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res4_scale2_bias, CL_TRUE, 0, 512, all_weights + 5415424, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res5_conv1_weight, CL_TRUE, 0, 589824, all_weights + 5415936, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res5_scale1_scale, CL_TRUE, 0, 512, all_weights + 6005760, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res5_scale1_bias, CL_TRUE, 0, 512, all_weights + 6006272, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res5_conv2_weight, CL_TRUE, 0, 589824, all_weights + 6006784, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res5_scale2_scale, CL_TRUE, 0, 512, all_weights + 6596608, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res5_scale2_bias, CL_TRUE, 0, 512, all_weights + 6597120, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, deconv5_1_weight, CL_TRUE, 0, 524288, all_weights + 6597632, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, deconv5_1_bias, CL_TRUE, 0, 256, all_weights + 7121920, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, deconv5_1_bn_sc_scale, CL_TRUE, 0, 256, all_weights + 7122176, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, deconv5_1_bn_sc_bias, CL_TRUE, 0, 256, all_weights + 7122432, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, deconv5_2_weight, CL_TRUE, 0, 131072, all_weights + 7122688, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, deconv5_2_bias, CL_TRUE, 0, 128, all_weights + 7253760, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, deconv5_2_bn_sc_scale, CL_TRUE, 0, 128, all_weights + 7253888, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, deconv5_2_bn_sc_bias, CL_TRUE, 0, 128, all_weights + 7254016, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, deconv5_3_weight, CL_TRUE, 0, 31104, all_weights + 7254144, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, deconv5_3_bias, CL_TRUE, 0, 12, all_weights + 7285248, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, image_scale1_scale, CL_TRUE, 0, 12, all_weights + 7285260, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, image_scale1_bias, CL_TRUE, 0, 12, all_weights + 7285272, 0, NULL, NULL));
OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, image_scale2_scale, CL_TRUE, 0, 12, all_weights + 7285284, 0, NULL, NULL));
free(all_weights);



OpenCLHandler::Get().build_opencl_program(conv_opencl_funcs, OpenCLHandler::Get().conv_program);
}


~new_net() {

OPENCL_CHECK(clReleaseMemObject(conv1_weight));
OPENCL_CHECK(clReleaseMemObject(conv1_bias));
OPENCL_CHECK(clReleaseMemObject(scale1_scale));
OPENCL_CHECK(clReleaseMemObject(scale1_bias));
OPENCL_CHECK(clReleaseMemObject(conv2_weight));
OPENCL_CHECK(clReleaseMemObject(conv2_bias));
OPENCL_CHECK(clReleaseMemObject(scale2_scale));
OPENCL_CHECK(clReleaseMemObject(scale2_bias));
OPENCL_CHECK(clReleaseMemObject(conv3_weight));
OPENCL_CHECK(clReleaseMemObject(conv3_bias));
OPENCL_CHECK(clReleaseMemObject(scale3_scale));
OPENCL_CHECK(clReleaseMemObject(scale3_bias));
OPENCL_CHECK(clReleaseMemObject(res1_conv1_weight));
OPENCL_CHECK(clReleaseMemObject(res1_scale1_scale));
OPENCL_CHECK(clReleaseMemObject(res1_scale1_bias));
OPENCL_CHECK(clReleaseMemObject(res1_conv2_weight));
OPENCL_CHECK(clReleaseMemObject(res1_scale2_scale));
OPENCL_CHECK(clReleaseMemObject(res1_scale2_bias));
OPENCL_CHECK(clReleaseMemObject(res2_conv1_weight));
OPENCL_CHECK(clReleaseMemObject(res2_scale1_scale));
OPENCL_CHECK(clReleaseMemObject(res2_scale1_bias));
OPENCL_CHECK(clReleaseMemObject(res2_conv2_weight));
OPENCL_CHECK(clReleaseMemObject(res2_scale2_scale));
OPENCL_CHECK(clReleaseMemObject(res2_scale2_bias));
OPENCL_CHECK(clReleaseMemObject(res3_conv1_weight));
OPENCL_CHECK(clReleaseMemObject(res3_scale1_scale));
OPENCL_CHECK(clReleaseMemObject(res3_scale1_bias));
OPENCL_CHECK(clReleaseMemObject(res3_conv2_weight));
OPENCL_CHECK(clReleaseMemObject(res3_scale2_scale));
OPENCL_CHECK(clReleaseMemObject(res3_scale2_bias));
OPENCL_CHECK(clReleaseMemObject(res4_conv1_weight));
OPENCL_CHECK(clReleaseMemObject(res4_scale1_scale));
OPENCL_CHECK(clReleaseMemObject(res4_scale1_bias));
OPENCL_CHECK(clReleaseMemObject(res4_conv2_weight));
OPENCL_CHECK(clReleaseMemObject(res4_scale2_scale));
OPENCL_CHECK(clReleaseMemObject(res4_scale2_bias));
OPENCL_CHECK(clReleaseMemObject(res5_conv1_weight));
OPENCL_CHECK(clReleaseMemObject(res5_scale1_scale));
OPENCL_CHECK(clReleaseMemObject(res5_scale1_bias));
OPENCL_CHECK(clReleaseMemObject(res5_conv2_weight));
OPENCL_CHECK(clReleaseMemObject(res5_scale2_scale));
OPENCL_CHECK(clReleaseMemObject(res5_scale2_bias));
OPENCL_CHECK(clReleaseMemObject(deconv5_1_weight));
OPENCL_CHECK(clReleaseMemObject(deconv5_1_bias));
OPENCL_CHECK(clReleaseMemObject(deconv5_1_bn_sc_scale));
OPENCL_CHECK(clReleaseMemObject(deconv5_1_bn_sc_bias));
OPENCL_CHECK(clReleaseMemObject(deconv5_2_weight));
OPENCL_CHECK(clReleaseMemObject(deconv5_2_bias));
OPENCL_CHECK(clReleaseMemObject(deconv5_2_bn_sc_scale));
OPENCL_CHECK(clReleaseMemObject(deconv5_2_bn_sc_bias));
OPENCL_CHECK(clReleaseMemObject(deconv5_3_weight));
OPENCL_CHECK(clReleaseMemObject(deconv5_3_bias));
OPENCL_CHECK(clReleaseMemObject(image_scale1_scale));
OPENCL_CHECK(clReleaseMemObject(image_scale1_bias));
OPENCL_CHECK(clReleaseMemObject(image_scale2_scale));

}
void inference( std::vector<float> &data_from_user, std::vector<float> &deconv5_3_to_user) { 


cl_mem conv1_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 33554432, NULL, NULL);
cl_mem conv2_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 16777216, NULL, NULL);
cl_mem conv3_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem conv3_scale3_0_split_0_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem conv3_scale3_0_split_1_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem data_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 3145728, NULL, NULL);
cl_mem deconv5_1_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 16777216, NULL, NULL);
cl_mem deconv5_2_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 33554432, NULL, NULL);
cl_mem deconv5_3_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 3145728, NULL, NULL);
cl_mem res1_conv1_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem res1_conv2_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem res1_output_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem res1_output_res1_elewise_0_split_0_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem res1_output_res1_elewise_0_split_1_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem res2_conv1_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem res2_conv2_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem res2_output_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem res2_output_res2_elewise_0_split_0_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem res2_output_res2_elewise_0_split_1_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem res3_conv1_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem res3_conv2_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem res3_output_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem res3_output_res3_elewise_0_split_0_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem res3_output_res3_elewise_0_split_1_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem res4_conv1_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem res4_conv2_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem res4_output_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem res4_output_res4_elewise_0_split_0_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem res4_output_res4_elewise_0_split_1_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem res5_conv1_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem res5_conv2_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);
cl_mem res5_output_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, 8388608, NULL, NULL);


OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, data_data, CL_TRUE, 0, data_from_user.size() * sizeof(data_from_user[0]), data_from_user.data(), 0, NULL, NULL));


conv1.Forward({data_data},{conv1_data});
ELU1.Forward(conv1_data,conv1_data);
bn1.Forward(conv1_data,conv1_data);
scale1.Forward(conv1_data,conv1_data);
conv2.Forward({conv1_data},{conv2_data});
ELU2.Forward(conv2_data,conv2_data);
bn2.Forward(conv2_data,conv2_data);
scale2.Forward(conv2_data,conv2_data);
conv3.Forward({conv2_data},{conv3_data});
ELU3.Forward(conv3_data,conv3_data);
bn3.Forward(conv3_data,conv3_data);
scale3.Forward(conv3_data,conv3_data);
conv3_scale3_0_split.Forward(conv3_data,{conv3_scale3_0_split_0_data , conv3_scale3_0_split_1_data});
res1_conv1.Forward({conv3_scale3_0_split_0_data},{res1_conv1_data});
res1_bn1.Forward(res1_conv1_data,res1_conv1_data);
res1_scale1.Forward(res1_conv1_data,res1_conv1_data);
res1_ReLU1.Forward(res1_conv1_data,res1_conv1_data);
res1_conv2.Forward({res1_conv1_data},{res1_conv2_data});
res1_bn2.Forward(res1_conv2_data,res1_conv2_data);
res1_scale2.Forward(res1_conv2_data,res1_conv2_data);
res1_elewise.Forward({res1_conv2_data , conv3_scale3_0_split_1_data},res1_output_data);
res1_output_res1_elewise_0_split.Forward(res1_output_data,{res1_output_res1_elewise_0_split_0_data , res1_output_res1_elewise_0_split_1_data});
res2_conv1.Forward({res1_output_res1_elewise_0_split_0_data},{res2_conv1_data});
res2_bn1.Forward(res2_conv1_data,res2_conv1_data);
res2_scale1.Forward(res2_conv1_data,res2_conv1_data);
res2_ReLU1.Forward(res2_conv1_data,res2_conv1_data);
res2_conv2.Forward({res2_conv1_data},{res2_conv2_data});
res2_bn2.Forward(res2_conv2_data,res2_conv2_data);
res2_scale2.Forward(res2_conv2_data,res2_conv2_data);
res2_elewise.Forward({res2_conv2_data , res1_output_res1_elewise_0_split_1_data},res2_output_data);
res2_output_res2_elewise_0_split.Forward(res2_output_data,{res2_output_res2_elewise_0_split_0_data , res2_output_res2_elewise_0_split_1_data});
res3_conv1.Forward({res2_output_res2_elewise_0_split_0_data},{res3_conv1_data});
res3_bn1.Forward(res3_conv1_data,res3_conv1_data);
res3_scale1.Forward(res3_conv1_data,res3_conv1_data);
res3_ReLU1.Forward(res3_conv1_data,res3_conv1_data);
res3_conv2.Forward({res3_conv1_data},{res3_conv2_data});
res3_bn2.Forward(res3_conv2_data,res3_conv2_data);
res3_scale2.Forward(res3_conv2_data,res3_conv2_data);
res3_elewise.Forward({res3_conv2_data , res2_output_res2_elewise_0_split_1_data},res3_output_data);
res3_output_res3_elewise_0_split.Forward(res3_output_data,{res3_output_res3_elewise_0_split_0_data , res3_output_res3_elewise_0_split_1_data});
res4_conv1.Forward({res3_output_res3_elewise_0_split_0_data},{res4_conv1_data});
res4_bn1.Forward(res4_conv1_data,res4_conv1_data);
res4_scale1.Forward(res4_conv1_data,res4_conv1_data);
res4_ReLU1.Forward(res4_conv1_data,res4_conv1_data);
res4_conv2.Forward({res4_conv1_data},{res4_conv2_data});
res4_bn2.Forward(res4_conv2_data,res4_conv2_data);
res4_scale2.Forward(res4_conv2_data,res4_conv2_data);
res4_elewise.Forward({res4_conv2_data , res3_output_res3_elewise_0_split_1_data},res4_output_data);
res4_output_res4_elewise_0_split.Forward(res4_output_data,{res4_output_res4_elewise_0_split_0_data , res4_output_res4_elewise_0_split_1_data});
res5_conv1.Forward({res4_output_res4_elewise_0_split_0_data},{res5_conv1_data});
res5_bn1.Forward(res5_conv1_data,res5_conv1_data);
res5_scale1.Forward(res5_conv1_data,res5_conv1_data);
res5_ReLU1.Forward(res5_conv1_data,res5_conv1_data);
res5_conv2.Forward({res5_conv1_data},{res5_conv2_data});
res5_bn2.Forward(res5_conv2_data,res5_conv2_data);
res5_scale2.Forward(res5_conv2_data,res5_conv2_data);
res5_elewise.Forward({res5_conv2_data , res4_output_res4_elewise_0_split_1_data},res5_output_data);
deconv5_1.Forward({res5_output_data},{deconv5_1_data});
deconv5_1_ELU.Forward(deconv5_1_data,deconv5_1_data);
deconv5_1_bn.Forward(deconv5_1_data,deconv5_1_data);
deconv5_1_bn_sc.Forward(deconv5_1_data,deconv5_1_data);
deconv5_2.Forward({deconv5_1_data},{deconv5_2_data});
deconv5_2_ELU.Forward(deconv5_2_data,deconv5_2_data);
deconv5_2_bn.Forward(deconv5_2_data,deconv5_2_data);
deconv5_2_bn_sc.Forward(deconv5_2_data,deconv5_2_data);
deconv5_3.Forward({deconv5_2_data},{deconv5_3_data});
tanh.Forward(deconv5_3_data,deconv5_3_data);
image_scale1.Forward(deconv5_3_data,deconv5_3_data);
image_scale2.Forward(deconv5_3_data,deconv5_3_data);


OPENCL_CHECK(clEnqueueReadBuffer(OpenCLHandler::Get().commandQueue, deconv5_3_data, CL_TRUE, 0, deconv5_3_to_user.size() * sizeof(deconv5_3_to_user[0]), deconv5_3_to_user.data(), 0, NULL, NULL));


OPENCL_CHECK(clReleaseMemObject(conv1_data));
OPENCL_CHECK(clReleaseMemObject(conv2_data));
OPENCL_CHECK(clReleaseMemObject(conv3_data));
OPENCL_CHECK(clReleaseMemObject(conv3_scale3_0_split_0_data));
OPENCL_CHECK(clReleaseMemObject(conv3_scale3_0_split_1_data));
OPENCL_CHECK(clReleaseMemObject(data_data));
OPENCL_CHECK(clReleaseMemObject(deconv5_1_data));
OPENCL_CHECK(clReleaseMemObject(deconv5_2_data));
OPENCL_CHECK(clReleaseMemObject(deconv5_3_data));
OPENCL_CHECK(clReleaseMemObject(res1_conv1_data));
OPENCL_CHECK(clReleaseMemObject(res1_conv2_data));
OPENCL_CHECK(clReleaseMemObject(res1_output_data));
OPENCL_CHECK(clReleaseMemObject(res1_output_res1_elewise_0_split_0_data));
OPENCL_CHECK(clReleaseMemObject(res1_output_res1_elewise_0_split_1_data));
OPENCL_CHECK(clReleaseMemObject(res2_conv1_data));
OPENCL_CHECK(clReleaseMemObject(res2_conv2_data));
OPENCL_CHECK(clReleaseMemObject(res2_output_data));
OPENCL_CHECK(clReleaseMemObject(res2_output_res2_elewise_0_split_0_data));
OPENCL_CHECK(clReleaseMemObject(res2_output_res2_elewise_0_split_1_data));
OPENCL_CHECK(clReleaseMemObject(res3_conv1_data));
OPENCL_CHECK(clReleaseMemObject(res3_conv2_data));
OPENCL_CHECK(clReleaseMemObject(res3_output_data));
OPENCL_CHECK(clReleaseMemObject(res3_output_res3_elewise_0_split_0_data));
OPENCL_CHECK(clReleaseMemObject(res3_output_res3_elewise_0_split_1_data));
OPENCL_CHECK(clReleaseMemObject(res4_conv1_data));
OPENCL_CHECK(clReleaseMemObject(res4_conv2_data));
OPENCL_CHECK(clReleaseMemObject(res4_output_data));
OPENCL_CHECK(clReleaseMemObject(res4_output_res4_elewise_0_split_0_data));
OPENCL_CHECK(clReleaseMemObject(res4_output_res4_elewise_0_split_1_data));
OPENCL_CHECK(clReleaseMemObject(res5_conv1_data));
OPENCL_CHECK(clReleaseMemObject(res5_conv2_data));
OPENCL_CHECK(clReleaseMemObject(res5_output_data));


}


private:


cl_mem conv1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 31104, NULL, NULL);
cl_mem conv1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 128, NULL, NULL);
cl_mem scale1_scale = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 128, NULL, NULL);
cl_mem scale1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 128, NULL, NULL);
cl_mem conv2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 131072, NULL, NULL);
cl_mem conv2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
cl_mem scale2_scale = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
cl_mem scale2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
cl_mem conv3_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 524288, NULL, NULL);
cl_mem conv3_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
cl_mem scale3_scale = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
cl_mem scale3_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
cl_mem res1_conv1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 589824, NULL, NULL);
cl_mem res1_scale1_scale = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
cl_mem res1_scale1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
cl_mem res1_conv2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 589824, NULL, NULL);
cl_mem res1_scale2_scale = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
cl_mem res1_scale2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
cl_mem res2_conv1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 589824, NULL, NULL);
cl_mem res2_scale1_scale = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
cl_mem res2_scale1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
cl_mem res2_conv2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 589824, NULL, NULL);
cl_mem res2_scale2_scale = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
cl_mem res2_scale2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
cl_mem res3_conv1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 589824, NULL, NULL);
cl_mem res3_scale1_scale = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
cl_mem res3_scale1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
cl_mem res3_conv2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 589824, NULL, NULL);
cl_mem res3_scale2_scale = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
cl_mem res3_scale2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
cl_mem res4_conv1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 589824, NULL, NULL);
cl_mem res4_scale1_scale = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
cl_mem res4_scale1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
cl_mem res4_conv2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 589824, NULL, NULL);
cl_mem res4_scale2_scale = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
cl_mem res4_scale2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
cl_mem res5_conv1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 589824, NULL, NULL);
cl_mem res5_scale1_scale = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
cl_mem res5_scale1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
cl_mem res5_conv2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 589824, NULL, NULL);
cl_mem res5_scale2_scale = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
cl_mem res5_scale2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
cl_mem deconv5_1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 524288, NULL, NULL);
cl_mem deconv5_1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
cl_mem deconv5_1_bn_sc_scale = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
cl_mem deconv5_1_bn_sc_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
cl_mem deconv5_2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 131072, NULL, NULL);
cl_mem deconv5_2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 128, NULL, NULL);
cl_mem deconv5_2_bn_sc_scale = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 128, NULL, NULL);
cl_mem deconv5_2_bn_sc_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 128, NULL, NULL);
cl_mem deconv5_3_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 31104, NULL, NULL);
cl_mem deconv5_3_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 12, NULL, NULL);
cl_mem image_scale1_scale = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 12, NULL, NULL);
cl_mem image_scale1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 12, NULL, NULL);
cl_mem image_scale2_scale = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 12, NULL, NULL);


ConvolutionOp_GPU<float> conv1 = ConvolutionOp_GPU<float>( "conv1_forward", 1, conv1_weight, conv1_bias, std::vector<int> {16, 4, 1}, std::vector<int> {32768, 8, 1});
ELUOp_GPU<float> ELU1 = ELUOp_GPU<float>( 8388608, 1);
BatchNormOp_GPU<float> bn1 = BatchNormOp_GPU<float>( 8388608, 1, 32, 1e-05, 1, false, NULL, NULL);
ScaleOp_GPU<float> scale1 = ScaleOp_GPU<float>( 8388608, scale1_bias, scale1_scale, 32, 262144);
ConvolutionOp_GPU<float> conv2 = ConvolutionOp_GPU<float>( "conv2_forward", 1, conv2_weight, conv2_bias, std::vector<int> {16, 4, 1}, std::vector<int> {8192, 16, 1});
ELUOp_GPU<float> ELU2 = ELUOp_GPU<float>( 4194304, 1);
BatchNormOp_GPU<float> bn2 = BatchNormOp_GPU<float>( 4194304, 1, 64, 1e-05, 1, false, NULL, NULL);
ScaleOp_GPU<float> scale2 = ScaleOp_GPU<float>( 4194304, scale2_bias, scale2_scale, 64, 65536);
ConvolutionOp_GPU<float> conv3 = ConvolutionOp_GPU<float>( "conv3_forward", 1, conv3_weight, conv3_bias, std::vector<int> {16, 4, 1}, std::vector<int> {2048, 32, 1});
ELUOp_GPU<float> ELU3 = ELUOp_GPU<float>( 2097152, 1);
BatchNormOp_GPU<float> bn3 = BatchNormOp_GPU<float>( 2097152, 1, 128, 1e-05, 1, false, NULL, NULL);
ScaleOp_GPU<float> scale3 = ScaleOp_GPU<float>( 2097152, scale3_bias, scale3_scale, 128, 16384);
SplitOp_GPU<float> conv3_scale3_0_split = SplitOp_GPU<float>( 2097152);
ConvolutionOp_GPU<float> res1_conv1 = ConvolutionOp_GPU<float>( "res1_conv1_forward", 1, res1_conv1_weight, NULL, std::vector<int> {16, 4, 1}, std::vector<int> {2048, 32, 1});
BatchNormOp_GPU<float> res1_bn1 = BatchNormOp_GPU<float>( 2097152, 1, 128, 1e-05, 1, false, NULL, NULL);
ScaleOp_GPU<float> res1_scale1 = ScaleOp_GPU<float>( 2097152, res1_scale1_bias, res1_scale1_scale, 128, 16384);
ReLUOp_GPU<float> res1_ReLU1 = ReLUOp_GPU<float>( 2097152, 0);
ConvolutionOp_GPU<float> res1_conv2 = ConvolutionOp_GPU<float>( "res1_conv2_forward", 1, res1_conv2_weight, NULL, std::vector<int> {16, 4, 1}, std::vector<int> {2048, 32, 1});
BatchNormOp_GPU<float> res1_bn2 = BatchNormOp_GPU<float>( 2097152, 1, 128, 1e-05, 1, false, NULL, NULL);
ScaleOp_GPU<float> res1_scale2 = ScaleOp_GPU<float>( 2097152, res1_scale2_bias, res1_scale2_scale, 128, 16384);
EltwiseOp_GPU<float> res1_elewise = EltwiseOp_GPU<float>( 2097152, 2, hypertea::EltwiseParameter_EltwiseOp_SUM, NULL, std::vector<float> { 1, 1 });
SplitOp_GPU<float> res1_output_res1_elewise_0_split = SplitOp_GPU<float>( 2097152);
ConvolutionOp_GPU<float> res2_conv1 = ConvolutionOp_GPU<float>( "res2_conv1_forward", 1, res2_conv1_weight, NULL, std::vector<int> {16, 4, 1}, std::vector<int> {2048, 32, 1});
BatchNormOp_GPU<float> res2_bn1 = BatchNormOp_GPU<float>( 2097152, 1, 128, 1e-05, 1, false, NULL, NULL);
ScaleOp_GPU<float> res2_scale1 = ScaleOp_GPU<float>( 2097152, res2_scale1_bias, res2_scale1_scale, 128, 16384);
ReLUOp_GPU<float> res2_ReLU1 = ReLUOp_GPU<float>( 2097152, 0);
ConvolutionOp_GPU<float> res2_conv2 = ConvolutionOp_GPU<float>( "res2_conv2_forward", 1, res2_conv2_weight, NULL, std::vector<int> {16, 4, 1}, std::vector<int> {2048, 32, 1});
BatchNormOp_GPU<float> res2_bn2 = BatchNormOp_GPU<float>( 2097152, 1, 128, 1e-05, 1, false, NULL, NULL);
ScaleOp_GPU<float> res2_scale2 = ScaleOp_GPU<float>( 2097152, res2_scale2_bias, res2_scale2_scale, 128, 16384);
EltwiseOp_GPU<float> res2_elewise = EltwiseOp_GPU<float>( 2097152, 2, hypertea::EltwiseParameter_EltwiseOp_SUM, NULL, std::vector<float> { 1, 1 });
SplitOp_GPU<float> res2_output_res2_elewise_0_split = SplitOp_GPU<float>( 2097152);
ConvolutionOp_GPU<float> res3_conv1 = ConvolutionOp_GPU<float>( "res3_conv1_forward", 1, res3_conv1_weight, NULL, std::vector<int> {16, 4, 1}, std::vector<int> {2048, 32, 1});
BatchNormOp_GPU<float> res3_bn1 = BatchNormOp_GPU<float>( 2097152, 1, 128, 1e-05, 1, false, NULL, NULL);
ScaleOp_GPU<float> res3_scale1 = ScaleOp_GPU<float>( 2097152, res3_scale1_bias, res3_scale1_scale, 128, 16384);
ReLUOp_GPU<float> res3_ReLU1 = ReLUOp_GPU<float>( 2097152, 0);
ConvolutionOp_GPU<float> res3_conv2 = ConvolutionOp_GPU<float>( "res3_conv2_forward", 1, res3_conv2_weight, NULL, std::vector<int> {16, 4, 1}, std::vector<int> {2048, 32, 1});
BatchNormOp_GPU<float> res3_bn2 = BatchNormOp_GPU<float>( 2097152, 1, 128, 1e-05, 1, false, NULL, NULL);
ScaleOp_GPU<float> res3_scale2 = ScaleOp_GPU<float>( 2097152, res3_scale2_bias, res3_scale2_scale, 128, 16384);
EltwiseOp_GPU<float> res3_elewise = EltwiseOp_GPU<float>( 2097152, 2, hypertea::EltwiseParameter_EltwiseOp_SUM, NULL, std::vector<float> { 1, 1 });
SplitOp_GPU<float> res3_output_res3_elewise_0_split = SplitOp_GPU<float>( 2097152);
ConvolutionOp_GPU<float> res4_conv1 = ConvolutionOp_GPU<float>( "res4_conv1_forward", 1, res4_conv1_weight, NULL, std::vector<int> {16, 4, 1}, std::vector<int> {2048, 32, 1});
BatchNormOp_GPU<float> res4_bn1 = BatchNormOp_GPU<float>( 2097152, 1, 128, 1e-05, 1, false, NULL, NULL);
ScaleOp_GPU<float> res4_scale1 = ScaleOp_GPU<float>( 2097152, res4_scale1_bias, res4_scale1_scale, 128, 16384);
ReLUOp_GPU<float> res4_ReLU1 = ReLUOp_GPU<float>( 2097152, 0);
ConvolutionOp_GPU<float> res4_conv2 = ConvolutionOp_GPU<float>( "res4_conv2_forward", 1, res4_conv2_weight, NULL, std::vector<int> {16, 4, 1}, std::vector<int> {2048, 32, 1});
BatchNormOp_GPU<float> res4_bn2 = BatchNormOp_GPU<float>( 2097152, 1, 128, 1e-05, 1, false, NULL, NULL);
ScaleOp_GPU<float> res4_scale2 = ScaleOp_GPU<float>( 2097152, res4_scale2_bias, res4_scale2_scale, 128, 16384);
EltwiseOp_GPU<float> res4_elewise = EltwiseOp_GPU<float>( 2097152, 2, hypertea::EltwiseParameter_EltwiseOp_SUM, NULL, std::vector<float> { 1, 1 });
SplitOp_GPU<float> res4_output_res4_elewise_0_split = SplitOp_GPU<float>( 2097152);
ConvolutionOp_GPU<float> res5_conv1 = ConvolutionOp_GPU<float>( "res5_conv1_forward", 1, res5_conv1_weight, NULL, std::vector<int> {16, 4, 1}, std::vector<int> {2048, 32, 1});
BatchNormOp_GPU<float> res5_bn1 = BatchNormOp_GPU<float>( 2097152, 1, 128, 1e-05, 1, false, NULL, NULL);
ScaleOp_GPU<float> res5_scale1 = ScaleOp_GPU<float>( 2097152, res5_scale1_bias, res5_scale1_scale, 128, 16384);
ReLUOp_GPU<float> res5_ReLU1 = ReLUOp_GPU<float>( 2097152, 0);
ConvolutionOp_GPU<float> res5_conv2 = ConvolutionOp_GPU<float>( "res5_conv2_forward", 1, res5_conv2_weight, NULL, std::vector<int> {16, 4, 1}, std::vector<int> {2048, 32, 1});
BatchNormOp_GPU<float> res5_bn2 = BatchNormOp_GPU<float>( 2097152, 1, 128, 1e-05, 1, false, NULL, NULL);
ScaleOp_GPU<float> res5_scale2 = ScaleOp_GPU<float>( 2097152, res5_scale2_bias, res5_scale2_scale, 128, 16384);
EltwiseOp_GPU<float> res5_elewise = EltwiseOp_GPU<float>( 2097152, 2, hypertea::EltwiseParameter_EltwiseOp_SUM, NULL, std::vector<float> { 1, 1 });
DeconvolutionOp_GPU<float> deconv5_1 = DeconvolutionOp_GPU<float>( "deconv5_1_forward", 1, deconv5_1_weight, deconv5_1_bias, std::vector<int> {16, 4, 1}, std::vector<int> {8192, 16, 1});
ELUOp_GPU<float> deconv5_1_ELU = ELUOp_GPU<float>( 4194304, 1);
BatchNormOp_GPU<float> deconv5_1_bn = BatchNormOp_GPU<float>( 4194304, 1, 64, 1e-05, 1, false, NULL, NULL);
ScaleOp_GPU<float> deconv5_1_bn_sc = ScaleOp_GPU<float>( 4194304, deconv5_1_bn_sc_bias, deconv5_1_bn_sc_scale, 64, 65536);
DeconvolutionOp_GPU<float> deconv5_2 = DeconvolutionOp_GPU<float>( "deconv5_2_forward", 1, deconv5_2_weight, deconv5_2_bias, std::vector<int> {16, 4, 1}, std::vector<int> {32768, 8, 1});
ELUOp_GPU<float> deconv5_2_ELU = ELUOp_GPU<float>( 8388608, 1);
BatchNormOp_GPU<float> deconv5_2_bn = BatchNormOp_GPU<float>( 8388608, 1, 32, 1e-05, 1, false, NULL, NULL);
ScaleOp_GPU<float> deconv5_2_bn_sc = ScaleOp_GPU<float>( 8388608, deconv5_2_bn_sc_bias, deconv5_2_bn_sc_scale, 32, 262144);
DeconvolutionOp_GPU<float> deconv5_3 = DeconvolutionOp_GPU<float>( "deconv5_3_forward", 1, deconv5_3_weight, deconv5_3_bias, std::vector<int> {16, 4, 1}, std::vector<int> {32768, 4, 1});
TanHOp_GPU<float> tanh = TanHOp_GPU<float>( 786432);
ScaleOp_GPU<float> image_scale1 = ScaleOp_GPU<float>( 786432, image_scale1_bias, image_scale1_scale, 3, 262144);
ScaleOp_GPU<float> image_scale2 = ScaleOp_GPU<float>( 786432, NULL, image_scale2_scale, 3, 262144);


};
} //namespace hypertea

