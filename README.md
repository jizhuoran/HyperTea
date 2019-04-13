HyperTea (fork from caffe-android-opencl-fp16 with totally new design principle)
==================

现今，深度学习已不再拘泥于计算能力强大的工作站或者是数据中心，转而更多的部署到边缘计算平台上，例如手机和IoT设备。然而，绝大部分著名的深度学习框架是为炼丹机设计的。为了能让炼丹师们更方便的设计和训练神经网络，这些深度学习框架引入了巨大的非必要开支（overhead）。 在边缘设备上，我们并不需要这些方便，我们需要的是能够快速的加载神经网络并进行推理计算。

当然，现在也有一些芯片生产厂家发布了一些用作在边缘设备的神经网络推理引擎（比如某通），也有一些直接在操作系统里集成了神经网络，例如人脸解锁（某为，某果）。但是，这些引擎要么是闭源且设备绑定的，开发者们只能使用这些引擎提供的功能和操作，没有办法发挥自己的聪明才智，更没有办法建立可移植的深度学习应用。

如果我们不能使用这些神经网络，我们干嘛要训练这些神经网络。因为，我们设计了HyperTea，一个极其轻量级的，inference-only 的神经网络推理引擎。 HyperTea支持CPU和GPU，并且主要为移动端的GPU进行了优化。HyperTea的速度只能是说非常的快。（我会告诉你我没有系统的测？）


(Hope this English version can maintain the meaning of the Chinese one...) Nowadays, deep learning is not a privilege for powerful workstation and data centers. Even a mobile phone or a embedded IoT device can do neural network inference, locally. 


However, most of the famous deep learning framework are designed for powerful workstations rather than edged AI devices for training neural networks. The loading overhead of traditional deep learning framework is intolerably high. However, most of the overhead is due to the need for neural network designing and training. 


We admit that some SoC manufacture release close-source deep neural network inference engines or has neural networks integrated in their operation systems (such as face recolonization in your phones). However, application developer can not benefits from these OS integrated neural networks nor extend these engines as they are not open-sourced.


If we do not have a tool to execute the neural network, why we bother to train them. Therefore, we designed HyperTea, an extremely light-weight Forward-Only deep neural network inference engine with both CPU and GPU supporting.

**I am developing this project. You can watch this project if you want to get the latest news**

# Features

- [x] Extreme fast loading and constriction of the neural network.

- [x] Designed and **optimized** for less powerful devices, such as mobile phones and embedded IoT chips.

- [x] Good Software Engineering, at least I think so. 

- [x] The neural network need to be defined in a hpp file, and the parameters (such as the input dim and output dim of the linear operator) are hard-coded into the constructor. **However, you never need to do it yourself, we provide model converter for Caffe and Pytorch now. Tensorflow is going to be supported, lattttter.**

- [x] The weights are stored as a single byte array, and the address is hard-coded into the network definition file. During loading, only a single byte array needs to be loaded from the disk.

- [x] As a Forward-Only neural network engine, you can not train neural networks with HyperTea. But don't worry, you can convert your pre-trained model into a HyperTea-Net and enjoy the excited inference speed.



# TODO
 
- [] A Doc, I promise...

- [] After refactoring, the core features (fast convolution operation and batch normalization operation designed for mobile GPU) are temporary unavailable. It will be fixed soon. (Okay, soon)

- [] The generator is just a S\*\*TING prototype. It will be released once refactoring is finished. You can try HyperTea with the example net.

- [] The test has not been done yet (I know I have promised it for a long time...)



# For Android

```
$ git clone https://github.com/jizhuoran/HyperTea.git
$ ./build_android.sh #you need to change the line 3 and line 4 of build_android.sh to your correspoing address
```

# For Ubuntu

## Test Environment

CPU: Intel(R) Xeon(R) CPU E5-2630 v4  
GPU NVIDIA 2080
OS: ubuntu 16.04  
OpenCL Version: 1.2  
C++ Version: 5.4.0  


## Step 1: Install CPU-Inference dependency

```
$ sudo apt-get install libopenblas-dev # Ubuntu
```

## Step 2: Install GPU-Inference dependency
```
$ git clone https://github.com/CNugteren/CLBlast.git
$ mkdir build && cd build && cmake .. && make -j40 && sudo make install
```

## Step 3: Build hypertea with cmake

```
$ git clone https://github.com/jizhuoran/HyperTea.git
$ mkdir build
$ cd ../build
$ cmake ..
$ make -j40
```

## Step 3: Build Test with cmake

```
$ git clone https://github.com/jizhuoran/HyperTea.git
$ mkdir build
$ cd ../build
$ cmake ..
$ make -j40 run test
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

# What is HyperTea

Hypertea is a kind of bottle tea drink sold in Mainland China. It reflect nothing about this neural network inference engine, it's my girlfriend's favorite bottle-drink~~, but to be honest, I do not think there is something special of that drink. As a tea drink, it contains sugar...~~


