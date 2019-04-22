#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <chrono>

#include "demo_net.hpp"
// #include "../ppm_reader.hpp"


#ifdef USE_OPENCL
using Timer = hypertea::GPUTimer;
using DeviceTensor = hypertea::TensorGPU<float>;
#else
using Timer = hypertea::CPUTimer;
using DeviceTensor = hypertea::TensorCPU<float>;
#endif

int main(int argc, char** argv) {



    std::vector<int> input_vector(1*25, 0);

    std::cout << "The size of the tensor is " << input_vector.size();

    std::vector<float> output_vector(1);



    hypertea::AttenNet<DeviceTensor> poem_net("/home/zrji/hypertea/examples/atten/pytorch_weight");


    Timer timer;

    timer.Start();
    
    for (int i = 0; i < 100; ++i) {
        std::cout << "This is the i th time " << std::endl;

        poem_net.inference(input_vector, output_vector);
    }
    
    timer.Stop();

    std::cout << "Time difference = " << timer.MilliSeconds() << "ms" <<std::endl;
    

    for (auto const&x: output_vector) {
        std::cout << x << " " << std::endl;
    }
    std::cout << " " << std::endl;


    
}