#include "hypertea/hypertea.hpp"
#include "bn_opencl_half.hpp"
#include "conv_opencl_half.hpp"

namespace hypertea {

class new_net {
public:

    new_net(const std::string &param_file) {

        int weight_size = 3642630;
        unsigned char* all_weights = (unsigned char*) malloc(weight_size);

        FILE *f = fopen(param_file.c_str(), "rb");
        size_t read_size = fread(all_weights, 1, weight_size, f);
        if (read_size != weight_size) { 
            LOG(ERROR) << "Weight File Size Mismatch" << read_size << " and " << weight_size << std::endl;
        }
        fclose(f);

        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, conv1_bias, CL_TRUE, 0, 64, all_weights + 0, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, conv1_weight, CL_TRUE, 0, 15552, all_weights + 64, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, bn1_weight, CL_TRUE, 0, 64, all_weights + 15616, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, bn1_bias, CL_TRUE, 0, 64, all_weights + 15680, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, conv2_bias, CL_TRUE, 0, 128, all_weights + 15744, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, conv2_weight, CL_TRUE, 0, 65536, all_weights + 15872, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, bn2_weight, CL_TRUE, 0, 128, all_weights + 81408, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, bn2_bias, CL_TRUE, 0, 128, all_weights + 81536, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, conv3_bias, CL_TRUE, 0, 256, all_weights + 81664, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, conv3_weight, CL_TRUE, 0, 262144, all_weights + 81920, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, bn3_weight, CL_TRUE, 0, 256, all_weights + 344064, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, bn3_bias, CL_TRUE, 0, 256, all_weights + 344320, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res1_conv1_weight, CL_TRUE, 0, 294912, all_weights + 344576, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res1_bn1_weight, CL_TRUE, 0, 256, all_weights + 639488, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res1_bn1_bias, CL_TRUE, 0, 256, all_weights + 639744, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res1_conv2_weight, CL_TRUE, 0, 294912, all_weights + 640000, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res1_bn2_weight, CL_TRUE, 0, 256, all_weights + 934912, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res1_bn2_bias, CL_TRUE, 0, 256, all_weights + 935168, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res2_conv1_weight, CL_TRUE, 0, 294912, all_weights + 935424, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res2_bn1_weight, CL_TRUE, 0, 256, all_weights + 1230336, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res2_bn1_bias, CL_TRUE, 0, 256, all_weights + 1230592, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res2_conv2_weight, CL_TRUE, 0, 294912, all_weights + 1230848, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res2_bn2_weight, CL_TRUE, 0, 256, all_weights + 1525760, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res2_bn2_bias, CL_TRUE, 0, 256, all_weights + 1526016, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res3_conv1_weight, CL_TRUE, 0, 294912, all_weights + 1526272, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res3_bn1_weight, CL_TRUE, 0, 256, all_weights + 1821184, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res3_bn1_bias, CL_TRUE, 0, 256, all_weights + 1821440, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res3_conv2_weight, CL_TRUE, 0, 294912, all_weights + 1821696, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res3_bn2_weight, CL_TRUE, 0, 256, all_weights + 2116608, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res3_bn2_bias, CL_TRUE, 0, 256, all_weights + 2116864, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res4_conv1_weight, CL_TRUE, 0, 294912, all_weights + 2117120, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res4_bn1_weight, CL_TRUE, 0, 256, all_weights + 2412032, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res4_bn1_bias, CL_TRUE, 0, 256, all_weights + 2412288, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res4_conv2_weight, CL_TRUE, 0, 294912, all_weights + 2412544, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res4_bn2_weight, CL_TRUE, 0, 256, all_weights + 2707456, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res4_bn2_bias, CL_TRUE, 0, 256, all_weights + 2707712, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res5_conv1_weight, CL_TRUE, 0, 294912, all_weights + 2707968, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res5_bn1_weight, CL_TRUE, 0, 256, all_weights + 3002880, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res5_bn1_bias, CL_TRUE, 0, 256, all_weights + 3003136, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res5_conv2_weight, CL_TRUE, 0, 294912, all_weights + 3003392, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res5_bn2_weight, CL_TRUE, 0, 256, all_weights + 3298304, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, res5_bn2_bias, CL_TRUE, 0, 256, all_weights + 3298560, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, deconv1_bias, CL_TRUE, 0, 128, all_weights + 3298816, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, deconv1_weight, CL_TRUE, 0, 262144, all_weights + 3298944, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, de_bn1_weight, CL_TRUE, 0, 128, all_weights + 3561088, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, de_bn1_bias, CL_TRUE, 0, 128, all_weights + 3561216, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, deconv2_bias, CL_TRUE, 0, 64, all_weights + 3561344, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, deconv2_weight, CL_TRUE, 0, 65536, all_weights + 3561408, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, de_bn2_weight, CL_TRUE, 0, 64, all_weights + 3626944, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, de_bn2_bias, CL_TRUE, 0, 64, all_weights + 3627008, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, deconv3_bias, CL_TRUE, 0, 6, all_weights + 3627072, 0, NULL, NULL));
        OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, deconv3_weight, CL_TRUE, 0, 15552, all_weights + 3627078, 0, NULL, NULL));

        free(all_weights);

        OpenCLHandler::Get().build_opencl_math_code(true);
        OpenCLHandler::Get().build_opencl_program(conv_opencl_funcs, OpenCLHandler::Get().conv_program);
        OpenCLHandler::Get().build_opencl_program(bn_opencl_funcs, OpenCLHandler::Get().bn_program);

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

    
    
    void inference( std::vector<half> &data_from_user, std::vector<half> &data_to_user) {
        
        TensorGPU<half> data(data_from_user);

        auto temp = conv1(data);

        float* debug_temp = new float[temp.count()];

        temp = bn1(elu1(temp));
        temp = bn2(elu2(conv2(temp)));
        temp = bn3(elu3(conv3(temp)));


        temp += res1_bn2(res1_conv2(res1_relu1(res1_bn1(res1_conv1(temp)))));
        temp += res2_bn2(res2_conv2(res2_relu1(res2_bn1(res2_conv1(temp)))));
        temp += res3_bn2(res3_conv2(res3_relu1(res3_bn1(res3_conv1(temp)))));
        temp += res4_bn2(res4_conv2(res4_relu1(res4_bn1(res4_conv1(temp)))));
        temp += res5_bn2(res5_conv2(res5_relu1(res5_bn1(res5_conv1(temp)))));
        

        temp = de_bn1(de_elu1(deconv1(temp)));
        temp = de_bn2(de_elu2(deconv2(temp)));
        temp = de_tanh3(deconv3(temp));

        temp = (temp + 1) * 127.5;

        OPENCL_CHECK(clEnqueueReadBuffer(OpenCLHandler::Get().commandQueue, temp.immutable_data(), CL_TRUE, 0, data_to_user.size() * sizeof(data_to_user[0]), data_to_user.data(), 0, NULL, NULL));

    }


private:

     

    cl_mem conv1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 64, NULL, NULL);
    cl_mem conv1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 15552, NULL, NULL);
    cl_mem bn1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 64, NULL, NULL);
    cl_mem bn1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 64, NULL, NULL);
    cl_mem conv2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 128, NULL, NULL);
    cl_mem conv2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 65536, NULL, NULL);
    cl_mem bn2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 128, NULL, NULL);
    cl_mem bn2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 128, NULL, NULL);
    cl_mem conv3_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem conv3_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 262144, NULL, NULL);
    cl_mem bn3_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem bn3_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem res1_conv1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 294912, NULL, NULL);
    cl_mem res1_bn1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem res1_bn1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem res1_conv2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 294912, NULL, NULL);
    cl_mem res1_bn2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem res1_bn2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem res2_conv1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 294912, NULL, NULL);
    cl_mem res2_bn1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem res2_bn1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem res2_conv2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 294912, NULL, NULL);
    cl_mem res2_bn2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem res2_bn2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem res3_conv1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 294912, NULL, NULL);
    cl_mem res3_bn1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem res3_bn1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem res3_conv2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 294912, NULL, NULL);
    cl_mem res3_bn2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem res3_bn2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem res4_conv1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 294912, NULL, NULL);
    cl_mem res4_bn1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem res4_bn1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem res4_conv2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 294912, NULL, NULL);
    cl_mem res4_bn2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem res4_bn2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem res5_conv1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 294912, NULL, NULL);
    cl_mem res5_bn1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem res5_bn1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem res5_conv2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 294912, NULL, NULL);
    cl_mem res5_bn2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem res5_bn2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 256, NULL, NULL);
    cl_mem deconv1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 128, NULL, NULL);
    cl_mem deconv1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 262144, NULL, NULL);
    cl_mem de_bn1_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 128, NULL, NULL);
    cl_mem de_bn1_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 128, NULL, NULL);
    cl_mem deconv2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 64, NULL, NULL);
    cl_mem deconv2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 65536, NULL, NULL);
    cl_mem de_bn2_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 64, NULL, NULL);
    cl_mem de_bn2_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 64, NULL, NULL);
    cl_mem deconv3_bias = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 6, NULL, NULL);
    cl_mem deconv3_weight = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, 15552, NULL, NULL);


	ConvolutionOp_GPU<half> conv1 = ConvolutionOp_GPU<half> ("conv1_forward", 8388608, conv1_weight, conv1_bias, std::vector<int> {16,4,1}, std::vector<int> {32768,8,1});
    ELUOp_GPU<half> elu1 = ELUOp_GPU<half> ( 1, NOT_IN_PLACE );
    BatchNormOp_GPU<half> bn1 = BatchNormOp_GPU<half> (8388608, 1, 32, 1e-05, 1, false, NULL, NULL, bn1_weight, bn1_bias, 64.0, 32.0);
    ConvolutionOp_GPU<half> conv2 = ConvolutionOp_GPU<half> ("conv2_forward", 4194304, conv2_weight, conv2_bias, std::vector<int> {16,4,1}, std::vector<int> {8192,16,1});
    ELUOp_GPU<half> elu2 = ELUOp_GPU<half> ( 1, NOT_IN_PLACE );
    BatchNormOp_GPU<half> bn2 = BatchNormOp_GPU<half> (4194304, 1, 64, 1e-05, 1, false, NULL, NULL, bn2_weight, bn2_bias, 64.0, 32.0);
    ConvolutionOp_GPU<half> conv3 = ConvolutionOp_GPU<half> ("conv3_forward", 2097152, conv3_weight, conv3_bias, std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    ELUOp_GPU<half> elu3 = ELUOp_GPU<half> ( 1, NOT_IN_PLACE );
    BatchNormOp_GPU<half> bn3 = BatchNormOp_GPU<half> (2097152, 1, 128, 1e-05, 1, false, NULL, NULL, bn3_weight, bn3_bias, 64.0, 32.0);
    ConvolutionOp_GPU<half> res1_conv1 = ConvolutionOp_GPU<half> ("res1_conv1_forward", 2097152, res1_conv1_weight, NULL, std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<half> res1_bn1 = BatchNormOp_GPU<half> (2097152, 1, 128, 1e-05, 1, false, NULL, NULL, res1_bn1_weight, res1_bn1_bias, 64.0, 32.0);
    ReLUOp_GPU<half> res1_relu1 = ReLUOp_GPU<half> ( 0, NOT_IN_PLACE );
    ConvolutionOp_GPU<half> res1_conv2 = ConvolutionOp_GPU<half> ("res1_conv2_forward", 2097152, res1_conv2_weight, NULL, std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<half> res1_bn2 = BatchNormOp_GPU<half> (2097152, 1, 128, 1e-05, 1, false, NULL, NULL, res1_bn2_weight, res1_bn2_bias, 64.0, 32.0);
    ConvolutionOp_GPU<half> res2_conv1 = ConvolutionOp_GPU<half> ("res2_conv1_forward", 2097152, res2_conv1_weight, NULL, std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<half> res2_bn1 = BatchNormOp_GPU<half> (2097152, 1, 128, 1e-05, 1, false, NULL, NULL, res2_bn1_weight, res2_bn1_bias, 64.0, 32.0);
    ReLUOp_GPU<half> res2_relu1 = ReLUOp_GPU<half> ( 0, NOT_IN_PLACE );
    ConvolutionOp_GPU<half> res2_conv2 = ConvolutionOp_GPU<half> ("res2_conv2_forward", 2097152, res2_conv2_weight, NULL, std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<half> res2_bn2 = BatchNormOp_GPU<half> (2097152, 1, 128, 1e-05, 1, false, NULL, NULL, res2_bn2_weight, res2_bn2_bias, 64.0, 32.0);
    ConvolutionOp_GPU<half> res3_conv1 = ConvolutionOp_GPU<half> ("res3_conv1_forward", 2097152, res3_conv1_weight, NULL, std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<half> res3_bn1 = BatchNormOp_GPU<half> (2097152, 1, 128, 1e-05, 1, false, NULL, NULL, res3_bn1_weight, res3_bn1_bias, 64.0, 32.0);
    ReLUOp_GPU<half> res3_relu1 = ReLUOp_GPU<half> ( 0, NOT_IN_PLACE );
    ConvolutionOp_GPU<half> res3_conv2 = ConvolutionOp_GPU<half> ("res3_conv2_forward", 2097152, res3_conv2_weight, NULL, std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<half> res3_bn2 = BatchNormOp_GPU<half> (2097152, 1, 128, 1e-05, 1, false, NULL, NULL, res3_bn2_weight, res3_bn2_bias, 64.0, 32.0);
    ConvolutionOp_GPU<half> res4_conv1 = ConvolutionOp_GPU<half> ("res4_conv1_forward", 2097152, res4_conv1_weight, NULL, std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<half> res4_bn1 = BatchNormOp_GPU<half> (2097152, 1, 128, 1e-05, 1, false, NULL, NULL, res4_bn1_weight, res4_bn1_bias, 64.0, 32.0);
    ReLUOp_GPU<half> res4_relu1 = ReLUOp_GPU<half> ( 0, NOT_IN_PLACE );
    ConvolutionOp_GPU<half> res4_conv2 = ConvolutionOp_GPU<half> ("res4_conv2_forward", 2097152, res4_conv2_weight, NULL, std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<half> res4_bn2 = BatchNormOp_GPU<half> (2097152, 1, 128, 1e-05, 1, false, NULL, NULL, res4_bn2_weight, res4_bn2_bias, 64.0, 32.0);
    ConvolutionOp_GPU<half> res5_conv1 = ConvolutionOp_GPU<half> ("res5_conv1_forward", 2097152, res5_conv1_weight, NULL, std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<half> res5_bn1 = BatchNormOp_GPU<half> (2097152, 1, 128, 1e-05, 1, false, NULL, NULL, res5_bn1_weight, res5_bn1_bias, 64.0, 32.0);
    ReLUOp_GPU<half> res5_relu1 = ReLUOp_GPU<half> ( 0, NOT_IN_PLACE );
    ConvolutionOp_GPU<half> res5_conv2 = ConvolutionOp_GPU<half> ("res5_conv2_forward", 2097152, res5_conv2_weight, NULL, std::vector<int> {16,4,1}, std::vector<int> {2048,32,1});
    BatchNormOp_GPU<half> res5_bn2 = BatchNormOp_GPU<half> (2097152, 1, 128, 1e-05, 1, false, NULL, NULL, res5_bn2_weight, res5_bn2_bias, 64.0, 32.0);
    DeconvolutionOp_GPU<half> deconv1 = DeconvolutionOp_GPU<half> ("deconv1_forward", 4194304, deconv1_weight, deconv1_bias, std::vector<int> {16,4,1}, std::vector<int> {8192,16,1});
    ELUOp_GPU<half> de_elu1 = ELUOp_GPU<half> ( 1, NOT_IN_PLACE );
    BatchNormOp_GPU<half> de_bn1 = BatchNormOp_GPU<half> (4194304, 1, 64, 1e-05, 1, false, NULL, NULL, de_bn1_weight, de_bn1_bias, 64.0, 32.0);
    DeconvolutionOp_GPU<half> deconv2 = DeconvolutionOp_GPU<half> ("deconv2_forward", 8388608, deconv2_weight, deconv2_bias, std::vector<int> {16,4,1}, std::vector<int> {32768,8,1});
    ELUOp_GPU<half> de_elu2 = ELUOp_GPU<half> ( 1, NOT_IN_PLACE );
    BatchNormOp_GPU<half> de_bn2 = BatchNormOp_GPU<half> (8388608, 1, 32, 1e-05, 1, false, NULL, NULL, de_bn2_weight, de_bn2_bias, 64.0, 32.0);
    DeconvolutionOp_GPU<half> deconv3 = DeconvolutionOp_GPU<half> ("deconv3_forward", 786432, deconv3_weight, deconv3_bias, std::vector<int> {16,4,1}, std::vector<int> {32768,4,1});
    TanHOp_GPU<half> de_tanh3 = TanHOp_GPU<half> ( NOT_IN_PLACE );


};

 

} //namespace hypertea
        
