#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <chrono>

#include "demo_net.hpp"
#include "../ppm_reader.hpp"


#ifdef USE_OPENCL
using Timer = hypertea::GPUTimer;
using DeviceTensor = hypertea::TensorGPU<int16>;
#else
using Timer = hypertea::CPUTimer;
using DeviceTensor = hypertea::TensorCPU<float>;
#endif

int main(int argc, char** argv) {


    PPMImage *image;
    image = readPPM("./examples/style_transfer/HKU.ppm");

    std::vector<int16> converter(512*512*3, 0);
    std::vector<int16> converter1(512*512*3, 0);

    for (int y = 0; y < 512; y++) {
      for (int x = 0; x < 512; x++) {
        converter[y * 512 + x] = image->data[y * 512 + x].red;
        converter[y * 512 + x + 512 * 512] = image->data[y * 512 + x].green;
        converter[y * 512 + x + 2 * 512 * 512] = image->data[y * 512 + x].blue;

      }
    }

    


    hypertea::new_net<DeviceTensor> style_transfer_net("./tools/style_transfer/pytorch_weight");


    Timer timer;

    timer.Start();
    style_transfer_net.inference(converter, converter1);
    timer.Stop();

    std::cout << "Time difference = " << timer.MilliSeconds() << "ms" <<std::endl;
    




    FILE *f = fopen("./examples/style_transfer/hypertea.ppm", "wb");
    fprintf(f, "P6\n%i %i 255\n", 512, 512);
    for (int y = 0; y < 512; y++) {
        for (int x = 0; x < 512; x++) {
            fputc(converter1[y * 512 + x], f);   // 0 .. 255
            fputc(converter1[y * 512 + x + 512 * 512], f); // 0 .. 255
            fputc(converter1[y * 512 + x + 2 * 512 * 512], f);  // 0 .. 255
        }
    }
    fclose(f);




    
}