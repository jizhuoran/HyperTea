// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.
#ifndef TEST_TEST_HYPERTEA_UTIL_HPP_
#define TEST_TEST_HYPERTEA_UTIL_HPP_


#include "hypertea/glog_wrapper.hpp"

#include <gtest/gtest.h>

#include <cstdio> 
#include <cstdlib>

// #include "caffe/common.hpp"

// using std::cout;
// using std::endl;

#ifdef CMAKE_BUILD
  #include "caffe_config.h"
#else
  #define CUDA_TEST_DEVICE -1
  #define EXAMPLES_SOURCE_DIR "examples/"
  #define ABS_TEST_DATA_DIR "src/caffe/test/test_data"
#endif

int main(int argc, char** argv);

namespace hypertea {

#ifdef USE_OPENCL
typedef ::testing::Types<TensorCPU<float>, TensorGPU<float> > TestDtypes;
#else
typedef ::testing::Types<TensorCPU<float> > TestDtypes;
#endif //USE_OPENCL

class fake_random_number {
public:
  fake_random_number() {
    
    std::ifstream source;
    source.open("/home/zrji/hypertea_maker/random_number.txt", std::ios_base::in);

    float value;

    for (int i = 0; i < 64 * 1024; ++i) {
      source >> value;
      source_vec.push_back(value);
    }

  }

  ~fake_random_number() = default;


  std::vector<float> generate_random_vector(int value_nums) {

    std::vector<float> v;
    for (int i = 0; i < value_nums; ++i) {
      v.push_back(source_vec[pos]);
      pos = (pos + 1) % source_vec.size();
    }

    return v;
  }


  std::vector<float> source_vec;
  int pos = 0;
  
};

}  // namespace hypertea

#endif  // TEST_TEST_HYPERTEA_UTIL_HPP_
