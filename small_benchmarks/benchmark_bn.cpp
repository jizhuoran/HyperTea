#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>


#include "hypertea/common.hpp"
#include "hypertea/hypertea.hpp"
// #include "hypertea/operations/batch_norm_op.hpp"



class fake_random_number {
public:
  fake_random_number() {
    
    std::ifstream source;
    source.open("/home/zrji/hypertea_maker/random_number.txt", std::ios_base::in);

    float value;

    for (int i = 0; i < 64 * 1024; ++i) {
      source >> value;
      source_vec.push_back(value);
    }

  }

  ~fake_random_number() = default;


  std::vector<float> generate_random_vector(int value_nums) {

    std::vector<float> v;
    for (int i = 0; i < value_nums; ++i) {
      v.push_back(source_vec[pos]);
      pos = (pos + 1) % source_vec.size();
    }

    return v;
  }


  std::vector<float> source_vec;
  int pos = 0;
  
};



std::string bn1_opencl_funcs = R"(
#define Dtype float
#define Dtype4 float4


static inline void ReduceKernel(__local Dtype* lcl_mem,
                                unsigned int sum_stride,
                                unsigned int unit_id,
                                unsigned int unit_len)
{
    Dtype sum              = (Dtype)0.;
    unsigned int lcl_offset = unit_id * unit_len;

    for(unsigned int i = 0; i < unit_len; i += sum_stride) {
        sum += lcl_mem[lcl_offset + i];
    }
    lcl_mem[lcl_offset] = sum;
}
        
static inline void
bn1_forward_regLDSreduce(Dtype* value, __local Dtype* data, unsigned int localID, Dtype scale)
{
    data[localID] = *value;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 2))
        ReduceKernel(data, 1, localID, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 4))
        ReduceKernel(data, 4, localID, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 6))
        ReduceKernel(data, 16, localID, 64);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 8))
        ReduceKernel(data, 64, localID, 256);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID == 0)
        ReduceKernel(data, 256, localID, 1024);
    barrier(CLK_LOCAL_MEM_FENCE);
    *value = data[0] * scale;
}
        

__attribute__((reqd_work_group_size(1024, 1, 1))) __kernel void
bn1_forward_MIOpenBatchNormFwdTrainSpatial(const __global Dtype* __restrict in,
                               __global Dtype* __restrict out,
                               __constant Dtype* __restrict scale,
                               __constant Dtype* __restrict bias
                               ) {

    // SPATIAL

    Dtype mean        = (Dtype)0.;
    Dtype variance    = (Dtype)0.;
    Dtype invVariance = (Dtype)0.;
    Dtype pvscale, pvbias;

    __local Dtype lcl_bias;
    __local Dtype lcl_scale;

    uint index = 0;
    uint lid   = get_local_id(0);
    uint grpid = get_group_id(0);
    uint chwid = grpid * 262144;
    uint nidx  = 0;
    uint hwidx = 0;

    if(lid == 0) {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    Dtype4 read4;
    for(unsigned int k = lid << 2; k < 262144;
                                               k += 4096) {
        nidx  = k / 262144;
        hwidx = k - (nidx * 262144);
        index = nidx * 8388608 + chwid + hwidx;
        read4 = *((const global Dtype4*)(in + index));
        mean += (Dtype)read4.x;
        mean += (Dtype)read4.y;
        mean += (Dtype)read4.z;
        mean += (Dtype)read4.w;
        variance = mad((Dtype)read4.x, (Dtype)read4.x, variance);
        variance = mad((Dtype)read4.y, (Dtype)read4.y, variance);
        variance = mad((Dtype)read4.z, (Dtype)read4.z, variance);
        variance = mad((Dtype)read4.w, (Dtype)read4.w, variance);
    }
     

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
// REDUCE MEAN AND VARIANCE -----------------------

    local Dtype lcl_data[1024];
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);

    //bn1_forward_regLDSreduce(&mean, lcl_data, lid, (Dtype)3.814697265625e-06);

    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);


    //bn1_forward_regLDSreduce(&variance, lcl_data, lid, (Dtype)3.814697265625e-06);
    barrier(CLK_LOCAL_MEM_FENCE);

    // REDUCTION COMPLETE ---------------------------
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + 1e-05);

    pvscale = lcl_scale;
    pvbias  = lcl_bias;

    
    for(unsigned int k = lid; k < 262144; k += 1024) {
        nidx  = k / 262144;
        hwidx = k - (nidx * 262144);
        index = nidx * 8388608 + chwid + hwidx;
        out[index] =
            (Dtype)mad(pvscale, ((Dtype)(*(in + index)) - mean) * invVariance, pvbias);
    } // end for
        


} // end spatial norm
)";

int main(int argc, char** argv) {

  hypertea::OpenCLHandler::Get().build_opencl_math_code(false);
  hypertea::OpenCLHandler::Get().build_opencl_program(bn1_opencl_funcs, hypertea::OpenCLHandler::Get().bn_program);

  fake_random_number fg;


  hypertea::TensorGPU<float> a1(fg.generate_random_vector(18388608));
  hypertea::TensorGPU<float> a2(fg.generate_random_vector(18388608));
  
  // hypertea::TensorGPU<float> a3(18388608);
  // a3.copy_data(a1);


  hypertea::TensorGPU<float> bn1_weight(fg.generate_random_vector(32));
  hypertea::TensorGPU<float> bn1_bias(fg.generate_random_vector(32));


  hypertea::TensorGPU<float> bn2_weight(32);
  hypertea::TensorGPU<float> bn2_bias(32);

  bn2_weight.copy_data(bn1_weight);
  bn2_bias.copy_data(bn1_bias);

  
  auto bn1 = hypertea::BatchNormOp_GPU<float> (
    8388608, 1, 32, 1e-05, 1, false, NULL, NULL, 
    bn1_weight, bn1_bias
  );



  auto bn2 = hypertea::MIOpenBatchNormOp_GPU<float>(
    "bn1_forward",
    NULL, NULL,
    bn2_weight.mutable_data(), bn2_bias.mutable_data(),
    std::vector<size_t> {1024, 1, 1},
    std::vector<size_t> {32768, 1, 1},
    32
  );



  // auto timer = hypertea::GPUTimer();

  // timer.Start();

  // auto b = bn1(a1);

  // timer.Stop();




  // std::cout << "The time for native bn is " << timer.MilliSeconds() << std::endl;


  auto timer2 = hypertea::GPUTimer();

  timer2.Start();
  
  bn2(a1);

  timer2.Stop();


  std::cout << "The time for MIOpen bn is " << timer2.MilliSeconds() << std::endl;

  // auto c = bn2(a3);

  // auto b_data = b.debug_cpu_data();
  // // auto c_data = c.debug_cpu_data();

  // for (int i = 0; i < 10; ++i) {
  //   std::cout << b_data[i] << std::endl;
  // }

  // for (int i = 0; i < c.count(); ++i) {
  //   if (b_data[i] - c_data[i] < -1e-05 || b_data[i] - c_data[i] > 1e-05) {
  //     std::cout << b_data[i] << " " << c_data[i] << std::endl;
  //     exit(0);
  //   } 
  // }


  // int N = a.count();

  // float scalar_ = 5.0;
  // float scalar1_ = 7.0;

  

  // hypertea::opencl_launch_wrapper(
  //   hypertea::OpenCLHandler::Get().math_program,
  //   "scal_kernel",
  //   std::vector<std::pair<size_t, const void *> > {
  //     std::make_pair(sizeof(cl_mem), (void *)&a_data),
  //     std::make_pair(sizeof(float), (void *)&scalar_),
  //     std::make_pair(sizeof(cl_int), (void *)&N)
  //   },
  //   std::vector<size_t> {hypertea::HYPERTEA_GET_BLOCKS(N)},
  //   std::vector<size_t> {hypertea::HYPERTEA_OPENCL_NUM_THREADS}
  // );


  // timer.Start();
  // hypertea::opencl_launch_wrapper(
  //   hypertea::OpenCLHandler::Get().math_program,
  //   "outplace_scal_scalar_kernel",
  //   std::vector<std::pair<size_t, const void *> > {
  //     std::make_pair(sizeof(cl_mem), (void *)&b_data),
  //     std::make_pair(sizeof(cl_mem), (void *)&b_data),
  //     std::make_pair(sizeof(float), (void *)&scalar1_),
  //     std::make_pair(sizeof(cl_int), (void *)&N)
  //   },
  //   std::vector<size_t> {hypertea::HYPERTEA_GET_BLOCKS(N)},
  //   std::vector<size_t> {hypertea::HYPERTEA_OPENCL_NUM_THREADS}
  // );
  // timer.Stop();
  // std::cout << "The time for outplace op is " << timer.MilliSeconds() << std::endl;

  // int ia, ib, ic;

  // hypertea::TensorGPU<float> a(1024*1024*64);
  // hypertea::TensorGPU<float> b(1024*1024*64);

  // auto c = a+b;


  

  // hypertea::TensorGPU<float> d(1024*1024*64);

  // std::cout << reference_count(a.mutable_data()) << std::endl;

  // // hypertea::hypertea_gpu_add<float>(1024*1024*32, a1, b1, d.mutable_data());



  // std::cin >> ia;
  // std::cout << a.debug_cpu_data()[ia];

  // std::cin >> ib;
  // std::cout << b.debug_cpu_data()[ib];

  // auto a1 = a.sub_view(79, 1024*1024*32);
  // auto b1 = b.sub_view(101, 1024*1024*32);
  // auto c1 = a1 + b1;

  // std::cout << reference_count(a1.mutable_data()) << std::endl;


  // std::cin >> ib;


  // std::cout << c1.debug_cpu_data()[ib];
    
}