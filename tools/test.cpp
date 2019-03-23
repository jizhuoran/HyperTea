#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <chrono>

#include "hypertea/hypertea.hpp"




int main(int argc, char** argv) {

  std::vector<int> t1{32, 64};
  std::vector<int> t2{32, 64};

  std::cout << "Whether they are same" << (t1 == t2) << std::endl;

  std::vector< hypertea::TensorCPU<float> > _w_ih{hypertea::TensorCPU<float>(std::vector<int>{32, 64}), hypertea::TensorCPU<float>(std::vector<int>{32, 64}), hypertea::TensorCPU<float>(std::vector<int>{32, 64})};  
  std::vector< hypertea::TensorCPU<float> > _w_hh{hypertea::TensorCPU<float>(std::vector<int>{32, 64}), hypertea::TensorCPU<float>(std::vector<int>{32, 64}), hypertea::TensorCPU<float>(std::vector<int>{32, 64})};  
  std::vector< hypertea::TensorCPU<float> > _b_ih{hypertea::TensorCPU<float>(32), hypertea::TensorCPU<float>(32), hypertea::TensorCPU<float>(32)};  
  std::vector< hypertea::TensorCPU<float> > _b_hh{hypertea::TensorCPU<float>(32), hypertea::TensorCPU<float>(32), hypertea::TensorCPU<float>(32)};  

  std::cout << "The shape is " << _w_ih[0].shape()[0] << " and " << _w_ih[0].shape()[1] << std::endl;

  auto input_tensor = hypertea::TensorCPU<float>(64);
  auto hidden_tensor = hypertea::TensorCPU<float>(32);

  hypertea::GRUCell_CPU<float> gru;
  hypertea::CellParams<float> params(_w_ih, _w_hh, _b_ih, _b_hh);

  gru.Forward(input_tensor, hidden_tensor, params);



}
