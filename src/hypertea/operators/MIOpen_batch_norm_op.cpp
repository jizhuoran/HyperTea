#include <algorithm>
#include <vector>
#include <math.h>

#include "hypertea/operators/MIOpen_batch_norm_op.hpp"
#include "hypertea/util/math_functions.hpp"

namespace hypertea {

template <typename Dtype>
TensorGPU<Dtype> MIOpenBatchNormOp_GPU<Dtype>::Forward(TensorGPU<Dtype> input_tensor){

    const cl_mem input_data = input_tensor.immutable_data();
    TensorGPU<Dtype> output_tensor = TensorGPU<Dtype>(input_tensor.count());
    cl_mem output_data = output_tensor.mutable_data();



    if(single_)
    {
        
        cl_int ret;


        cl_kernel kernel = clCreateKernel(OpenCLHandler::Get().bn_program, (kernel_name_ + "_MIOpenBatchNormFwdTrainSpatial").c_str(), &ret);
        OPENCL_CHECK(ret);


          // Set arguments for kernel
        OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_data));  
        OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output_data));  
        OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&weight_));  
        OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bias_));   
        // OPENCL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_float), (void *)&inhw_));   
        // OPENCL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_float), (void *)&eps_));   
  

        size_t* global_size = global_size_.data();
        size_t* local_size  = local_size_.data();


        for (int i = 0; i < 1000; ++i) {
          OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, global_size, local_size, 0, NULL, NULL));  
        }
        


        
    }
    else
    {


        std::cout << "We need this part" << std::endl;
        exit(1);
        

        
    }


    return output_tensor;
}



INSTANTIATE_CLASS_GPU(MIOpenBatchNormOp_GPU);


}//namespace hypertea
