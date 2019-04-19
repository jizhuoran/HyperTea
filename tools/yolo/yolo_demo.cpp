#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <chrono>

#include "demo_net.hpp"
#include "../ppm_reader.hpp"


#ifdef USE_OPENCL
using Timer = hypertea::GPUTimer;
using DeviceTensor = hypertea::TensorGPU<float>;
#else
using Timer = hypertea::CPUTimer;
using DeviceTensor = hypertea::TensorCPU<float>;
#endif

int main(int argc, char** argv) {



    std::vector<float> input_vector(3*500*375);
    std::vector<float> output_vector(3*500*375);



    PPMImage *image;
    image = readPPM("/home/zrji/hypertea/examples/yolo/img4.ppm");


    for (int y = 0; y < 375; y++) {
      for (int x = 0; x < 500; x++) {
        input_vector[y * 500 + x] = image->data[y * 500 + x].red;
        input_vector[y * 500 + x + 500 * 500] = image->data[y * 500 + x].blue;
        input_vector[y * 500 + x + 2 * 500 * 500] = image->data[y * 500 + x].green;

      }
    }


    for (int i = 0; i < input_vector.size(); ++i) {
        input_vector[i] /= 255.0;
    }



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