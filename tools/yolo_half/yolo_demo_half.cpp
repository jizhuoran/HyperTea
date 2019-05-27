#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <chrono>

#include "demo_net.hpp"
#include "../ppm_reader.hpp"


#ifdef USE_OPENCL
using Timer = hypertea::GPUTimer;
using DeviceTensor = hypertea::TensorGPU<half>;
#else
using Timer = hypertea::CPUTimer;
using DeviceTensor = hypertea::TensorCPU<half>;
#endif

int main(int argc, char** argv) {



    std::vector<half> input_vector(3*416*416);
    std::vector<half> output_vector(1);



    PPMImage *image;
#ifdef __ANDROID__
    image = readPPM("/sdcard/hypertea_ws/yolo/img4.ppm");
#else
    image = readPPM("/home/zrji/hypertea/examples/yolo/img4.ppm");
#endif


    for (int y = 0; y < 416; y++) {
      for (int x = 0; x < 416; x++) {
        input_vector[y * 416 + x] = image->data[y * 416 + x].red;
        input_vector[y * 416 + x + 416 * 416] = image->data[y * 416 + x].green;
        input_vector[y * 416 + x + 2 * 416 * 416] = image->data[y * 416 + x].blue;

      }
    }


    for (int i = 0; i < input_vector.size(); ++i) {
        input_vector[i] /= 255.0;
    }


    hypertea::CPUTimer load_timer;

    load_timer.Start();

#ifdef __ANDROID__
    hypertea::yolo_net<DeviceTensor> yolo3("/sdcard/hypertea_ws/yolo_half/pytorch_weight_half");
#else
    hypertea::yolo_net<DeviceTensor> yolo3("/home/zrji/hypertea/examples/yolo_half/pytorch_weight_half");
#endif

    load_timer.Stop();

    std::cout << "LOAD Time difference = " << load_timer.MilliSeconds() << "ms" <<std::endl;


    Timer inference_timer;

    inference_timer.Start();
    
    for (int i = 0; i < 1; ++i) {
        yolo3.inference(input_vector, output_vector);
    }
    
    inference_timer.Stop();

    std::cout << "INFERENCE Time difference = " << inference_timer.MilliSeconds() << "ms" <<std::endl;
    

    // for (auto const&x: output_vector) {
    //     std::cout << x << " " << std::endl;
    // }
    // std::cout << " " << std::endl;


    
}