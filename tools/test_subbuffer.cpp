#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <chrono>

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

  int ia, ib, ic;

  hypertea::TensorGPU<float> a(1024*1024*64);
  hypertea::TensorGPU<float> b(1024*1024*64);

  auto c = a+b;


  

  hypertea::TensorGPU<float> d(1024*1024*64);

  std::cout << reference_count(a.mutable_data()) << std::endl;

  // hypertea::hypertea_gpu_add<float>(1024*1024*32, a1, b1, d.mutable_data());



  std::cin >> ia;


  std::cout << c.debug_cpu_data()[ia];

  auto a1 = a.sub_tensor_view(79, 1024*1024*32);
  auto b1 = b.sub_tensor_view(101, 1024*1024*32);
  auto c1 = a1 + b1;

  std::cout << reference_count(a1.mutable_data()) << std::endl;


  std::cin >> ib;


  std::cout << c1.debug_cpu_data()[ib];
    
}