
#include "hypertea/hypertea.hpp"

namespace hypertea {

class new_net {
public:

    new_net() {

        int weight_size = 7285260;
        unsigned char* all_weights = (unsigned char*) malloc(weight_size);

        FILE *f = fopen("pytorch_weight", "rb");
        size_t read_size = fread(all_weights, 1, weight_size, f);
        if (read_size != weight_size) { 
            LOG(ERROR) << "Weight File Size Mismatch" << read_size << " and " << weight_size << std::endl;
        }
        fclose(f);

        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, conv1_bias, CL_TRUE, 0, 128, all_weights + 0, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, conv1_weight, CL_TRUE, 0, 31104, all_weights + 128, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, bn1_weight, CL_TRUE, 0, 128, all_weights + 31232, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, bn1_bias, CL_TRUE, 0, 128, all_weights + 31360, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, conv2_bias, CL_TRUE, 0, 256, all_weights + 31488, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, conv2_weight, CL_TRUE, 0, 131072, all_weights + 31744, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, bn2_weight, CL_TRUE, 0, 256, all_weights + 162816, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, bn2_bias, CL_TRUE, 0, 256, all_weights + 163072, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, conv3_bias, CL_TRUE, 0, 512, all_weights + 163328, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, conv3_weight, CL_TRUE, 0, 524288, all_weights + 163840, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, bn3_weight, CL_TRUE, 0, 512, all_weights + 688128, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, bn3_bias, CL_TRUE, 0, 512, all_weights + 688640, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res1_conv1_weight, CL_TRUE, 0, 589824, all_weights + 689152, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res1_bn1_weight, CL_TRUE, 0, 512, all_weights + 1278976, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res1_bn1_bias, CL_TRUE, 0, 512, all_weights + 1279488, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res1_conv2_weight, CL_TRUE, 0, 589824, all_weights + 1280000, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res1_bn2_weight, CL_TRUE, 0, 512, all_weights + 1869824, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res1_bn2_bias, CL_TRUE, 0, 512, all_weights + 1870336, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res2_conv1_weight, CL_TRUE, 0, 589824, all_weights + 1870848, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res2_bn1_weight, CL_TRUE, 0, 512, all_weights + 2460672, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res2_bn1_bias, CL_TRUE, 0, 512, all_weights + 2461184, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res2_conv2_weight, CL_TRUE, 0, 589824, all_weights + 2461696, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res2_bn2_weight, CL_TRUE, 0, 512, all_weights + 3051520, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res2_bn2_bias, CL_TRUE, 0, 512, all_weights + 3052032, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res3_conv1_weight, CL_TRUE, 0, 589824, all_weights + 3052544, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res3_bn1_weight, CL_TRUE, 0, 512, all_weights + 3642368, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res3_bn1_bias, CL_TRUE, 0, 512, all_weights + 3642880, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res3_conv2_weight, CL_TRUE, 0, 589824, all_weights + 3643392, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res3_bn2_weight, CL_TRUE, 0, 512, all_weights + 4233216, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res3_bn2_bias, CL_TRUE, 0, 512, all_weights + 4233728, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res4_conv1_weight, CL_TRUE, 0, 589824, all_weights + 4234240, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res4_bn1_weight, CL_TRUE, 0, 512, all_weights + 4824064, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res4_bn1_bias, CL_TRUE, 0, 512, all_weights + 4824576, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res4_conv2_weight, CL_TRUE, 0, 589824, all_weights + 4825088, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res4_bn2_weight, CL_TRUE, 0, 512, all_weights + 5414912, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res4_bn2_bias, CL_TRUE, 0, 512, all_weights + 5415424, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res5_conv1_weight, CL_TRUE, 0, 589824, all_weights + 5415936, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res5_bn1_weight, CL_TRUE, 0, 512, all_weights + 6005760, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res5_bn1_bias, CL_TRUE, 0, 512, all_weights + 6006272, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res5_conv2_weight, CL_TRUE, 0, 589824, all_weights + 6006784, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res5_bn2_weight, CL_TRUE, 0, 512, all_weights + 6596608, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res5_bn2_bias, CL_TRUE, 0, 512, all_weights + 6597120, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, deconv1_bias, CL_TRUE, 0, 256, all_weights + 6597632, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, deconv1_weight, CL_TRUE, 0, 524288, all_weights + 6597888, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, de_bn1_weight, CL_TRUE, 0, 256, all_weights + 7122176, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, de_bn1_bias, CL_TRUE, 0, 256, all_weights + 7122432, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, deconv2_bias, CL_TRUE, 0, 128, all_weights + 7122688, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, deconv2_weight, CL_TRUE, 0, 131072, all_weights + 7122816, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, de_bn2_weight, CL_TRUE, 0, 128, all_weights + 7253888, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, de_bn2_bias, CL_TRUE, 0, 128, all_weights + 7254016, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, deconv3_bias, CL_TRUE, 0, 12, all_weights + 7254144, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, deconv3_weight, CL_TRUE, 0, 31104, all_weights + 7254156, 0, NULL, NULL));

        free(all_weights);
        // OpenCLHandler::Get().load_opencl_program("conv_cl", OpenCLHandler::Get().conv_program);
        // OpenCLHandler::Get().load_opencl_program("bn_cl", OpenCLHandler::Get().bn_program);
        OpenCLHandler::Get().build_opencl_program(conv_opencl_funcs, OpenCLHandler::Get().conv_program, "conv_cl");
        OpenCLHandler::Get().build_opencl_program(bn_opencl_funcs, OpenCLHandler::Get().bn_program, "bn_cl");

    }


    ~new_net() {
        OPENCL_CHECK(clReleaseMemObject(conv1_bias));
        OPENCL_CHECK(clReleaseMemObject(conv1_weight));
        OPENCL_CHECK(clReleaseMemObject(bn1_weight));
        OPENCL_CHECK(clReleaseMemObject(bn1_bias));
        OPENCL_CHECK(clReleaseMemObject(conv2_bias));
        OPENCL_CHECK(clReleaseMemObject(conv2_weight));
        OPENCL_CHECK(clReleaseMemObject(bn2_weight));
        OPENCL_CHECK(clReleaseMemObject(bn2_bias));
        OPENCL_CHECK(clReleaseMemObject(conv3_bias));
        OPENCL_CHECK(clReleaseMemObject(conv3_weight));
        OPENCL_CHECK(clReleaseMemObject(bn3_weight));
        OPENCL_CHECK(clReleaseMemObject(bn3_bias));
        OPENCL_CHECK(clReleaseMemObject(res1_conv1_weight));
        OPENCL_CHECK(clReleaseMemObject(res1_bn1_weight));
        OPENCL_CHECK(clReleaseMemObject(res1_bn1_bias));
        OPENCL_CHECK(clReleaseMemObject(res1_conv2_weight));
        OPENCL_CHECK(clReleaseMemObject(res1_bn2_weight));
        OPENCL_CHECK(clReleaseMemObject(res1_bn2_bias));
        OPENCL_CHECK(clReleaseMemObject(res2_conv1_weight));
        OPENCL_CHECK(clReleaseMemObject(res2_bn1_weight));
        OPENCL_CHECK(clReleaseMemObject(res2_bn1_bias));
        OPENCL_CHECK(clReleaseMemObject(res2_conv2_weight));
        OPENCL_CHECK(clReleaseMemObject(res2_bn2_weight));
        OPENCL_CHECK(clReleaseMemObject(res2_bn2_bias));
        OPENCL_CHECK(clReleaseMemObject(res3_conv1_weight));
        OPENCL_CHECK(clReleaseMemObject(res3_bn1_weight));
        OPENCL_CHECK(clReleaseMemObject(res3_bn1_bias));
        OPENCL_CHECK(clReleaseMemObject(res3_conv2_weight));
        OPENCL_CHECK(clReleaseMemObject(res3_bn2_weight));
        OPENCL_CHECK(clReleaseMemObject(res3_bn2_bias));
        OPENCL_CHECK(clReleaseMemObject(res4_conv1_weight));
        OPENCL_CHECK(clReleaseMemObject(res4_bn1_weight));
        OPENCL_CHECK(clReleaseMemObject(res4_bn1_bias));
        OPENCL_CHECK(clReleaseMemObject(res4_conv2_weight));
        OPENCL_CHECK(clReleaseMemObject(res4_bn2_weight));
        OPENCL_CHECK(clReleaseMemObject(res4_bn2_bias));
        OPENCL_CHECK(clReleaseMemObject(res5_conv1_weight));
        OPENCL_CHECK(clReleaseMemObject(res5_bn1_weight));
        OPENCL_CHECK(clReleaseMemObject(res5_bn1_bias));
        OPENCL_CHECK(clReleaseMemObject(res5_conv2_weight));
        OPENCL_CHECK(clReleaseMemObject(res5_bn2_weight));
        OPENCL_CHECK(clReleaseMemObject(res5_bn2_bias));
        OPENCL_CHECK(clReleaseMemObject(deconv1_bias));
        OPENCL_CHECK(clReleaseMemObject(deconv1_weight));
        OPENCL_CHECK(clReleaseMemObject(de_bn1_weight));
        OPENCL_CHECK(clReleaseMemObject(de_bn1_bias));
        OPENCL_CHECK(clReleaseMemObject(deconv2_bias));
        OPENCL_CHECK(clReleaseMemObject(deconv2_weight));
        OPENCL_CHECK(clReleaseMemObject(de_bn2_weight));
        OPENCL_CHECK(clReleaseMemObject(de_bn2_bias));
        OPENCL_CHECK(clReleaseMemObject(deconv3_bias));
        OPENCL_CHECK(clReleaseMemObject(deconv3_weight));
    }

    
    
    void inference( std::vector<float> &data_from_user, std::vector<float> &data_to_user) {
        
        TensorGPU<float> data(data_from_user);

        auto temp = bn1(elu1(conv1(data)));
        temp = bn2(elu2(conv2(temp)));
        temp = bn3(elu3(conv3(temp)));


        temp += res1_bn2(res1_conv2(res1_relu1(res1_bn1(res1_conv1(temp)))));
        temp += res2_bn2(res2_conv2(res2_relu1(res2_bn1(res2_conv1(temp)))));
        temp += res3_bn2(res3_conv2(res3_relu1(res3_bn1(res3_conv1(temp)))));
        temp += res4_bn2(res4_conv2(res4_relu1(res4_bn1(res4_conv1(temp)))));
        temp += res5_bn2(res5_conv2(res5_relu1(res5_bn1(res5_conv1(temp)))));
        

        float* print_data = temp.debug_cpu_data();
        for (int i = 0; i < 10; ++i) {
            std::cout << print_data[i] << " | "; 
        }


        temp = deconv1(temp);

        print_data = temp.debug_cpu_data();
        for (int i = 0; i < 10; ++i) {
            std::cout << print_data[i] << " | "; 
        }

        temp = de_bn1(de_elu1(temp));
        temp = de_bn2(de_elu2(deconv2(temp)));
        temp = de_tanh3(deconv3(temp));

        temp = (temp + 1) * 127.5;

        OPENCL_CHECK(clEnqueueReadBuffer(OpenCLHandler::Get().commandQueue, temp.immutable_data(), CL_TRUE, 0, data_to_user.size() * sizeof(data_to_user[0]), data_to_user.data(), 0, NULL, NULL));

    }


private:

    cl_mem conv1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 128, NULL, NULL);
    cl_mem conv1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 31104, NULL, NULL);
    cl_mem bn1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 128, NULL, NULL);
    cl_mem bn1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 128, NULL, NULL);
    cl_mem conv2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem conv2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 131072, NULL, NULL);
    cl_mem bn2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem bn2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem conv3_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
    cl_mem conv3_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 524288, NULL, NULL);
    cl_mem bn3_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
    cl_mem bn3_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
    cl_mem res1_conv1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 589824, NULL, NULL);
    cl_mem res1_bn1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
    cl_mem res1_bn1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
    cl_mem res1_conv2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 589824, NULL, NULL);
    cl_mem res1_bn2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
    cl_mem res1_bn2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
    cl_mem res2_conv1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 589824, NULL, NULL);
    cl_mem res2_bn1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
    cl_mem res2_bn1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
    cl_mem res2_conv2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 589824, NULL, NULL);
    cl_mem res2_bn2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
    cl_mem res2_bn2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
    cl_mem res3_conv1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 589824, NULL, NULL);
    cl_mem res3_bn1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
    cl_mem res3_bn1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
    cl_mem res3_conv2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 589824, NULL, NULL);
    cl_mem res3_bn2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
    cl_mem res3_bn2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
    cl_mem res4_conv1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 589824, NULL, NULL);
    cl_mem res4_bn1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
    cl_mem res4_bn1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
    cl_mem res4_conv2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 589824, NULL, NULL);
    cl_mem res4_bn2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
    cl_mem res4_bn2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
    cl_mem res5_conv1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 589824, NULL, NULL);
    cl_mem res5_bn1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
    cl_mem res5_bn1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
    cl_mem res5_conv2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 589824, NULL, NULL);
    cl_mem res5_bn2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
    cl_mem res5_bn2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 512, NULL, NULL);
    cl_mem deconv1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem deconv1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 524288, NULL, NULL);
    cl_mem de_bn1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem de_bn1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem deconv2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 128, NULL, NULL);
    cl_mem deconv2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 131072, NULL, NULL);
    cl_mem de_bn2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 128, NULL, NULL);
    cl_mem de_bn2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 128, NULL, NULL);
    cl_mem deconv3_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 12, NULL, NULL);
    cl_mem deconv3_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 31104, NULL, NULL);


    ConvolutionOp_GPU<float> conv1 = ConvolutionOp_GPU<float> ("conv1_forward", 8388608, conv1_weight, conv1_bias, std::vector<int> {16,4,1}, std::vector<int> {32768,8,1});
    ELUOp_GPU<float> elu1 = ELUOp_GPU<float> ( 1, NOT_IN_PLACE );
    BatchNormOp_GPU<float> bn1 = BatchNormOp_GPU<float> (8388608, 1, 32, 1e-05, 1, false, NULL, NULL, bn1_weight, bn1_bias);
    ConvolutionOp_GPU<float> conv2 = ConvolutionOp_GPU<float> ("conv2_forward", 4194304, conv2_weight, conv2_bias, std::vector<int> {16,4,1}, std::vector<int> {8192,16,1});
    ELUOp_GPU<float> elu2 = ELUOp_GPU<float> ( 1, NOT_IN_PLACE );
    BatchNormOp_GPU<float> bn2 = BatchNormOp_GPU<float> (4194304, 1, 64, 1e-05, 1, false, NULL, NULL, bn2_weight, bn2_bias);
    ConvolutionOp_GPU<float> conv3 = ConvolutionOp_GPU<float> ("conv3_forward", 2097152, conv3_weight, conv3_bias, std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    ELUOp_GPU<float> elu3 = ELUOp_GPU<float> ( 1, NOT_IN_PLACE );
    BatchNormOp_GPU<float> bn3 = BatchNormOp_GPU<float> (2097152, 1, 128, 1e-05, 1, false, NULL, NULL, bn3_weight, bn3_bias);
    ConvolutionOp_GPU<float> res1_conv1 = ConvolutionOp_GPU<float> ("res1_conv1_forward", 2097152, res1_conv1_weight, NULL, std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<float> res1_bn1 = BatchNormOp_GPU<float> (2097152, 1, 128, 1e-05, 1, false, NULL, NULL, res1_bn1_weight, res1_bn1_bias);
    ReLUOp_GPU<float> res1_relu1 = ReLUOp_GPU<float> ( 0, NOT_IN_PLACE );
    ConvolutionOp_GPU<float> res1_conv2 = ConvolutionOp_GPU<float> ("res1_conv2_forward", 2097152, res1_conv2_weight, NULL, std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<float> res1_bn2 = BatchNormOp_GPU<float> (2097152, 1, 128, 1e-05, 1, false, NULL, NULL, res1_bn2_weight, res1_bn2_bias);
    ConvolutionOp_GPU<float> res2_conv1 = ConvolutionOp_GPU<float> ("res2_conv1_forward", 2097152, res2_conv1_weight, NULL, std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<float> res2_bn1 = BatchNormOp_GPU<float> (2097152, 1, 128, 1e-05, 1, false, NULL, NULL, res2_bn1_weight, res2_bn1_bias);
    ReLUOp_GPU<float> res2_relu1 = ReLUOp_GPU<float> ( 0, NOT_IN_PLACE );
    ConvolutionOp_GPU<float> res2_conv2 = ConvolutionOp_GPU<float> ("res2_conv2_forward", 2097152, res2_conv2_weight, NULL, std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<float> res2_bn2 = BatchNormOp_GPU<float> (2097152, 1, 128, 1e-05, 1, false, NULL, NULL, res2_bn2_weight, res2_bn2_bias);
    ConvolutionOp_GPU<float> res3_conv1 = ConvolutionOp_GPU<float> ("res3_conv1_forward", 2097152, res3_conv1_weight, NULL, std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<float> res3_bn1 = BatchNormOp_GPU<float> (2097152, 1, 128, 1e-05, 1, false, NULL, NULL, res3_bn1_weight, res3_bn1_bias);
    ReLUOp_GPU<float> res3_relu1 = ReLUOp_GPU<float> ( 0, NOT_IN_PLACE );
    ConvolutionOp_GPU<float> res3_conv2 = ConvolutionOp_GPU<float> ("res3_conv2_forward", 2097152, res3_conv2_weight, NULL, std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<float> res3_bn2 = BatchNormOp_GPU<float> (2097152, 1, 128, 1e-05, 1, false, NULL, NULL, res3_bn2_weight, res3_bn2_bias);
    ConvolutionOp_GPU<float> res4_conv1 = ConvolutionOp_GPU<float> ("res4_conv1_forward", 2097152, res4_conv1_weight, NULL, std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<float> res4_bn1 = BatchNormOp_GPU<float> (2097152, 1, 128, 1e-05, 1, false, NULL, NULL, res4_bn1_weight, res4_bn1_bias);
    ReLUOp_GPU<float> res4_relu1 = ReLUOp_GPU<float> ( 0, NOT_IN_PLACE );
    ConvolutionOp_GPU<float> res4_conv2 = ConvolutionOp_GPU<float> ("res4_conv2_forward", 2097152, res4_conv2_weight, NULL, std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<float> res4_bn2 = BatchNormOp_GPU<float> (2097152, 1, 128, 1e-05, 1, false, NULL, NULL, res4_bn2_weight, res4_bn2_bias);
    ConvolutionOp_GPU<float> res5_conv1 = ConvolutionOp_GPU<float> ("res5_conv1_forward", 2097152, res5_conv1_weight, NULL, std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<float> res5_bn1 = BatchNormOp_GPU<float> (2097152, 1, 128, 1e-05, 1, false, NULL, NULL, res5_bn1_weight, res5_bn1_bias);
    ReLUOp_GPU<float> res5_relu1 = ReLUOp_GPU<float> ( 0, NOT_IN_PLACE );
    ConvolutionOp_GPU<float> res5_conv2 = ConvolutionOp_GPU<float> ("res5_conv2_forward", 2097152, res5_conv2_weight, NULL, std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<float> res5_bn2 = BatchNormOp_GPU<float> (2097152, 1, 128, 1e-05, 1, false, NULL, NULL, res5_bn2_weight, res5_bn2_bias);
    NativeDeconvolutionOp_GPU<float> deconv1 = NativeDeconvolutionOp_GPU<float> ("deconv1_col2Im", 4194304, 1024, deconv1_weight, deconv1_bias, std::vector<int> {1,128,128,128}, std::vector<int> {1,64,256,256}, false, std::vector<int> {256,1,1}, std::vector<int> {4194304,1,1});
    ELUOp_GPU<float> de_elu1 = ELUOp_GPU<float> ( 1, NOT_IN_PLACE );
    BatchNormOp_GPU<float> de_bn1 = BatchNormOp_GPU<float> (4194304, 1, 64, 1e-05, 1, false, NULL, NULL, de_bn1_weight, de_bn1_bias);
    NativeDeconvolutionOp_GPU<float> deconv2 = NativeDeconvolutionOp_GPU<float> ("deconv2_col2Im", 8388608, 512, deconv2_weight, deconv2_bias, std::vector<int> {1,64,256,256}, std::vector<int> {1,32,512,512}, false, std::vector<int> {256,1,1}, std::vector<int> {8388608,1,1});
    ELUOp_GPU<float> de_elu2 = ELUOp_GPU<float> ( 1, NOT_IN_PLACE );
    BatchNormOp_GPU<float> de_bn2 = BatchNormOp_GPU<float> (8388608, 1, 32, 1e-05, 1, false, NULL, NULL, de_bn2_weight, de_bn2_bias);
    NativeDeconvolutionOp_GPU<float> deconv3 = NativeDeconvolutionOp_GPU<float> ("deconv3_col2Im", 786432, 243, deconv3_weight, deconv3_bias, std::vector<int> {1,32,512,512}, std::vector<int> {1,3,512,512}, false, std::vector<int> {256,1,1}, std::vector<int> {786432,1,1});
    TanHOp_GPU<float> de_tanh3 = TanHOp_GPU<float> ( NOT_IN_PLACE );

};
} //namespace hypertea
        
