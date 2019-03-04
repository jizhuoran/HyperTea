#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <chrono>

#include "../tools/conv_opencl.hpp"
#include "../tools/demo_net_gpu.hpp"



#define MAX_SOURCE_SIZE (0x100000)

#define RGB_COMPONENT_COLOR 255

 
typedef struct {
     unsigned char red,green,blue;
} PPMPixel; 

typedef struct {
     int x, y;
     PPMPixel *data;
} PPMImage;


static PPMImage *readPPM(const char *filename) {
  char buff[16];
  FILE *fp;
  int c, rgb_comp_color;
  PPMImage *img;

 //open PPM file for reading
  
  fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open file '%s'\n", filename);
    exit(1);
  }

  //read image format
  if (!fgets(buff, sizeof(buff), fp)) {
    perror(filename);
    exit(1);
  }  

  //check the image format
  if (buff[0] != 'P' || buff[1] != '6') {
       fprintf(stderr, "Invalid image format (must be 'P6')\n");
       exit(1);
  }

  //alloc memory form image
  img = (PPMImage *)malloc(sizeof(PPMImage));
  if (!img) {
       fprintf(stderr, "Unable to allocate memory\n");
       exit(1);
  }

  //check for comments
  c = getc(fp);
  while (c == '#') {
  while (getc(fp) != '\n') ;
       c = getc(fp);
  }

  ungetc(c, fp);
  //read image size information
  if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
       fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
       exit(1);
  }

    //read rgb component
    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
         fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
         exit(1);
    }

    //check rgb component depth
    if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
         fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
         exit(1);
    }

    while (fgetc(fp) != '\n') ;
    //memory allocation for pixel data
    img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //read pixel data from file
    if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
         fprintf(stderr, "Error loading image '%s'\n", filename);
         exit(1);
    }
    fclose(fp);
    return img;
}



int main(int argc, char** argv) {



//     std::vector<float> temp_debug(10, 0);
// OPENCL_CHECK(clEnqueueReadBuffer(OpenCLHandler::Get().commandQueue, conv1_weight, CL_TRUE, 0, 40, temp_debug.data(), 0, NULL, NULL));
// for (auto const& x: temp_debug) {
//   std::cout << x << " ";
// }



    PPMImage *image;
    image = readPPM("./examples/style_transfer/HKU.ppm");

    std::vector<float> converter(512*512*3*1, 0);
    std::vector<float> converter1(512*512*3*1, 0);

    for (int y = 0; y < 512; y++) {
      for (int x = 0; x < 512; x++) {
        converter[y * 512 + x] = image->data[y * 512 + x].red;
        converter[y * 512 + x + 512 * 512] = image->data[y * 512 + x].green;
        converter[y * 512 + x + 2 * 512 * 512] = image->data[y * 512 + x].blue;


        // converter[512*512*3 + y * 512 + x] = image->data[y * 512 + x].red;
        // converter[512*512*3 + y * 512 + x + 512 * 512] = image->data[y * 512 + x].green;
        // converter[512*512*3 + y * 512 + x + 2 * 512 * 512] = image->data[y * 512 + x].blue;


      }
    }

    
    



    hypertea::new_net tmp_net;


    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (int i = 0; i < 1; ++i) {
    	tmp_net.inference(converter, converter1);
    }
  
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;
    

    // for(int i = 0; i < 512*512*3*1; ++i) {
    //   converter1[i] *= 127.5;
    // }


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


    for (int j = 0; j < 3; ++j){
      for (int i = 0; i < 4; ++i) {
        std::cout << converter1[j * 512 * 512 + i] << " ";
      }
      std::cout << " " << std::endl;
    }
    
    // f = fopen("./examples/style_transfer/hypertea2.ppm", "wb");
    // fprintf(f, "P6\n%i %i 255\n", 512, 512);
    // for (int y = 0; y < 512; y++) {
    //     for (int x = 0; x < 512; x++) {
    //         fputc(converter1[512*512*3 + y * 512 + x], f);   // 0 .. 255
    //         fputc(converter1[512*512*3 + y * 512 + x + 512 * 512], f); // 0 .. 255
    //         fputc(converter1[512*512*3 + y * 512 + x + 2 * 512 * 512], f);  // 0 .. 255
    //     }
    // }
    // fclose(f);

    
}