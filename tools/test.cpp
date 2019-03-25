#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cstdlib>

#include "hypertea/hypertea.hpp"


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




int main(int argc, char** argv) {


  fake_random_number random_generator;

  // std::vector< hypertea::TensorCPU<float> > _w_ih{hypertea::TensorCPU<float>(random_generator.generate_random_vector(32*64), std::vector<int>{32, 64}), hypertea::TensorCPU<float>(random_generator.generate_random_vector(32*64), std::vector<int>{32, 64}), hypertea::TensorCPU<float>(random_generator.generate_random_vector(32*64), std::vector<int>{32, 64})};  
  // std::vector< hypertea::TensorCPU<float> > _w_hh{hypertea::TensorCPU<float>(random_generator.generate_random_vector(32*32), std::vector<int>{32, 32}), hypertea::TensorCPU<float>(random_generator.generate_random_vector(32*32), std::vector<int>{32, 32}), hypertea::TensorCPU<float>(random_generator.generate_random_vector(32*32), std::vector<int>{32, 32})};  
  // std::vector< hypertea::TensorCPU<float> > _b_ih{hypertea::TensorCPU<float>(random_generator.generate_random_vector(32), std::vector<int>{32}), hypertea::TensorCPU<float>(random_generator.generate_random_vector(32), std::vector<int>{32}), hypertea::TensorCPU<float>(random_generator.generate_random_vector(32), std::vector<int>{32})};  
  // std::vector< hypertea::TensorCPU<float> > _b_hh{hypertea::TensorCPU<float>(random_generator.generate_random_vector(32), std::vector<int>{32}), hypertea::TensorCPU<float>(random_generator.generate_random_vector(32), std::vector<int>{32}), hypertea::TensorCPU<float>(random_generator.generate_random_vector(32), std::vector<int>{32})};  

  auto _w_ih = random_generator.generate_random_vector(3*32*64);
  auto _w_hh = random_generator.generate_random_vector(3*32*32);
  auto _b_ih = random_generator.generate_random_vector(3*32);
  auto _b_hh = random_generator.generate_random_vector(3*32);


  auto input_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(5*64), std::vector<int>{64});
  auto hidden_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(32), std::vector<int>{32});
  auto output_tensor = hypertea::TensorCPU<float>(32);



  // hypertea::GRUCell_CPU<float> gru(64, 32, _w_ih.data(), _w_hh.data(), _b_ih.data(), _b_hh.data());
  hypertea::GRUOp_CPU<float> gru(1, 64, 32, _w_ih.data(), _w_hh.data(), _b_ih.data(), _b_hh.data());


  output_tensor = gru.Forward(
    input_tensor,
    hidden_tensor
  );

  // gru.Forward(
  //   input_tensor.mutable_data(),
  //   hidden_tensor.mutable_data(),
  //   output_tensor.mutable_data()
  // );

  const float* temp_data = output_tensor.immutable_data();

  for (int i = 0; i < output_tensor.count(); ++i) {
    std::cout << temp_data[i] << " ";
  }


  std::cout << " " << std::endl;
  std::cout << "The count is " << output_tensor.count() << std::endl;

}
