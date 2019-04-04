#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>


#include "hypertea/common.hpp"
#include "hypertea/hypertea.hpp" 



size_t reference_count(cl_mem obj) {

    cl_uint refer_count;

    OPENCL_CHECK(
      clGetMemObjectInfo (
        obj,
        CL_MEM_REFERENCE_COUNT,
        sizeof(cl_uint),
        &refer_count,
        nullptr
      )
    );

    return refer_count;
}


int main(int argc, char** argv) {

  hypertea::OpenCLHandler::Get().build_opencl_math_code(false);

  hypertea::TensorGPU<float> a(1024*1024*512);
  hypertea::TensorGPU<float> b(1024*1024*512);

  cl_mem a_data = a.mutable_data();
  cl_mem b_data = b.mutable_data();

  int N = a.count();

  float scalar_ = 5.0;
  float scalar1_ = 7.0;

  auto timer = hypertea::GPUTimer();

  timer.Start();

  hypertea::opencl_launch_wrapper(
    hypertea::OpenCLHandler::Get().math_program,
    "scal_kernel",
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&a_data),
      std::make_pair(sizeof(float), (void *)&scalar_),
      std::make_pair(sizeof(cl_int), (void *)&N)
    },
    std::vector<size_t> {hypertea::HYPERTEA_GET_BLOCKS(N)},
    std::vector<size_t> {hypertea::HYPERTEA_OPENCL_NUM_THREADS}
  );

  timer.Stop();

  std::cout << "The time for inplace op is " << timer.MilliSeconds() << std::endl;


  timer.Start();
  hypertea::opencl_launch_wrapper(
    hypertea::OpenCLHandler::Get().math_program,
    "outplace_scal_scalar_kernel",
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&b_data),
      std::make_pair(sizeof(cl_mem), (void *)&b_data),
      std::make_pair(sizeof(float), (void *)&scalar1_),
      std::make_pair(sizeof(cl_int), (void *)&N)
    },
    std::vector<size_t> {hypertea::HYPERTEA_GET_BLOCKS(N)},
    std::vector<size_t> {hypertea::HYPERTEA_OPENCL_NUM_THREADS}
  );
  timer.Stop();
  std::cout << "The time for outplace op is " << timer.MilliSeconds() << std::endl;

  int ia, ib, ic;

  // hypertea::TensorGPU<float> a(1024*1024*64);
  // hypertea::TensorGPU<float> b(1024*1024*64);

  // auto c = a+b;


  

  // hypertea::TensorGPU<float> d(1024*1024*64);

  // std::cout << reference_count(a.mutable_data()) << std::endl;

  // // hypertea::hypertea_gpu_add<float>(1024*1024*32, a1, b1, d.mutable_data());



  std::cin >> ia;
  std::cout << a.debug_cpu_data()[ia];

  std::cin >> ib;
  std::cout << b.debug_cpu_data()[ib];

  // auto a1 = a.sub_view(79, 1024*1024*32);
  // auto b1 = b.sub_view(101, 1024*1024*32);
  // auto c1 = a1 + b1;

  // std::cout << reference_count(a1.mutable_data()) << std::endl;


  // std::cin >> ib;


  // std::cout << c1.debug_cpu_data()[ib];
    
}