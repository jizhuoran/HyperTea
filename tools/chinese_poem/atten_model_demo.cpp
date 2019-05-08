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



    std::vector<int> input_vector(25, 1);
    input_vector[0] = 0;

    std::cout << "The size of the tensor is " << input_vector.size() << std::endl;

    std::cout << "pass this line 0" << std::endl;

    std::vector<int> output_vector(1);

    hypertea::CPUTimer load_timer;

    load_timer.Start();

#ifdef __ANDROID__
    hypertea::AttenNet<DeviceTensor> poem_net("/sdcard/hypertea_ws/atten/pytorch_weight");
#else
    hypertea::AttenNet<DeviceTensor> poem_net("/home/zrji/hypertea/examples/atten/pytorch_weight");
#endif

    load_timer.Stop();

    std::cout << "LOAD Time difference = " << load_timer.MilliSeconds() << "ms" <<std::endl;


    Timer inference_timer;

    inference_timer.Start();
    
    for (int i = 0; i < 1; ++i) {
        poem_net.inference(input_vector, output_vector);
    }
    
    inference_timer.Stop();


    std::cout << "INFERENCE Time difference = " << inference_timer.MilliSeconds() << "ms" <<std::endl;
    

    for (auto const&x: output_vector) {
        std::cout << x << " ";
    }
    std::cout << " " << std::endl;


    
}