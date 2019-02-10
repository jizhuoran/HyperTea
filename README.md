HyperTea (fork from caffe-android-opencl-fp16 with totally new design principle)
==================

The loading overhead of traditional deep learning framework is intolerably high. However, most of the overhead is due to the need for neural network designing and training. Therefore, we designed HyperTea, an extremely light-weight Forward-Only deep neural network inference engine with both CPU and GPU supporting.

**I am developing this project. You can watch this project if you want to get the latest news**

# Features

- [x] The neural network need to be defined in a cpp file, and the parameters (such as the input dim and output dim of the linear operator) are hard-coded into the constructor.

- [x] The weights are stored as a single byte array, and the address is hard-coded into the network definition file. During loading, only a single byte array needs to be loaded from the disk.

- [x] As a Forward-Only neural network engine, you can not train neural networks with HyperTea. But don't worry, we will provide deep learning model convert for Caffe, Pytorch, and Tensorflow. You can convert your pre-trained model into a HyperTea-Net and enjoy the excited inference speed.

- [x] However, the generator is just a S\*\*TING prototype. It will be released once refactoring is finished. You can try HyperTea with the example net.

- [x] The test has not been done yet and I am going to go home and enjoy Chinese New Year.



# For Android

**Release after Chinese New Year**. 

# For Ubuntu

## Test Environment

CPU: Intel(R) Xeon(R) CPU E5-2630 v4  
GPU NVIDIA 2080
OS: ubuntu 16.04  
OpenCL Version: 1.2  
C++ Version: 5.4.0  


## Step 1: Install CPU-Inference dependency

```
$ sudo apt install libatlas-dev # Ubuntu
```

## Step 2: Install GPU-Inference dependency
```
$ git clone https://github.com/CNugteren/CLBlast.git
$ mkdir build && cd build && cmake .. && make -j40 && sudo make install
```

## Step 3: Build Caffe-Mobile Lib with cmake

```
$ git clone https://github.com/jizhuoran/HyperTea.git
$ mkdir build
$ cd ../build
$ cmake ..
$ make -j 40
```

## Step 4: Run

```
$ cd $HyperTea Home
$ ./build/tools/hypertea_demo.bin
```

# Thanks

 - Based on https://github.com/solrex/caffe-mobile.git
 - Also Based on https://github.com/jizhuoran/caffe-android-opencl-fp16.git
 - Inspired by https://github.com/BVLC/caffe/tree/opencl
