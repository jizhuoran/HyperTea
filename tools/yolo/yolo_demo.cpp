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



    std::vector<float> input_vector(3*416*416, 0.23);

    std::cout << "The size of the tensor is " << input_vector.size();

    std::vector<int> output_vector(1);

 

    



    hypertea::yolo_net<DeviceTensor> yolo3("/home/zrji/hypertea/examples/yolo/pytorch_weight");


    Timer timer;

    timer.Start();
    
    for (int i = 0; i < 1; ++i) {
        yolo3.inference(input_vector, output_vector);
    }
    
    timer.Stop();

    std::cout << "Time difference = " << timer.MilliSeconds() << "ms" <<std::endl;
    

    for (auto const&x: output_vector) {
        std::cout << x << " " << std::endl;
    }
    std::cout << " " << std::endl;


    
}