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



    std::vector<float> input_vector(3*112*96, 0.23);

    std::cout << "The size of the tensor is " << input_vector.size();

    std::vector<int> output_vector(1);



    
hypertea::CPUTimer load_timer;

load_timer.Start();

#ifdef __ANDROID__
    hypertea::facenet<DeviceTensor> face_net("/sdcard/hypertea_ws/facenet/pytorch_weight");
#else
    hypertea::facenet<DeviceTensor> face_net("/home/zrji/hypertea/examples/facenet/pytorch_weight");
#endif

load_timer.Stop();

std::cout << "LOAD Time difference = " << load_timer.MilliSeconds() << "ms" <<std::endl;



    Timer timer;

    timer.Start();
    
    for (int i = 0; i < 1; ++i) {
        face_net.inference(input_vector, output_vector);
    }
    
    timer.Stop();

    std::cout << "Time difference = " << timer.MilliSeconds() << "ms" <<std::endl;
    

    for (auto const&x: output_vector) {
        std::cout << x << " " << std::endl;
    }
    std::cout << " " << std::endl;


    
}