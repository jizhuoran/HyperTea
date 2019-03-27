#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cstdlib>

#include "hypertea/hypertea.hpp"
#include "../test_result/rnn_result.hpp"


template <typename T>
bool pass_test(std::string name, std::vector<T> groudtruth, const T* result, const T threshold) {
  for (int i = 0; i < groudtruth.size(); ++i) {
    if (abs(groudtruth[i] - result[i]) > threshold) {
      std::cout << "The test case "<< name <<" do not pass! Because of abs(" 
                << groudtruth[i] << " - " << result[i] << ">" << threshold << std::endl;
      return false;
    }
  }
  std::cout << "The test case "<< name <<" pass!" << std::endl; 
  return true;
}


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




void test_uni_single_gru() {
    
  fake_random_number random_generator;

  auto _w_ih0 = random_generator.generate_random_vector(3*32*64);
  auto _w_hh0 = random_generator.generate_random_vector(3*32*32);
  auto _b_ih0 = random_generator.generate_random_vector(3*32);
  auto _b_hh0 = random_generator.generate_random_vector(3*32);


  auto input_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(5*64), std::vector<int>{64});
  auto hidden_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(1*32), std::vector<int>{32});
  auto output_tensor = hypertea::TensorCPU<float>(32);

  hypertea::StackedRNN<float> gru( std::vector<hypertea::RNNOp_CPU<float>* >{
      new hypertea::UnidirectionalRNN_CPU<float> (
        1, 64, 32, 
        _w_ih0.data(), _w_hh0.data(), 
        _b_ih0.data(), _b_hh0.data(), 
        hypertea::RNN_CELL_TYPE::GRU_CELL
      )
    }
  );


  output_tensor = gru.Forward(
    input_tensor,
    std::vector<hypertea::TensorCPU<float> > {hidden_tensor}
  );


  const float* temp_data = output_tensor.immutable_data();

  pass_test(std::string("test_uni_single_gru"), hypertea::test_result::uni_single_gru_result, temp_data, float(0.001));

}


void test_bi_single_gru() {
    
  fake_random_number random_generator;

  auto _w_ih0 = random_generator.generate_random_vector(3*32*64);
  auto _w_hh0 = random_generator.generate_random_vector(3*32*32);
  auto _b_ih0 = random_generator.generate_random_vector(3*32);
  auto _b_hh0 = random_generator.generate_random_vector(3*32);

  auto r_w_ih0 = random_generator.generate_random_vector(3*32*64);
  auto r_w_hh0 = random_generator.generate_random_vector(3*32*32);
  auto r_b_ih0 = random_generator.generate_random_vector(3*32);
  auto r_b_hh0 = random_generator.generate_random_vector(3*32);

  auto input_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(5*64), std::vector<int>{64});
  auto hidden_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(2*32), std::vector<int>{32});
  auto output_tensor = hypertea::TensorCPU<float>(32);

  hypertea::StackedRNN<float> gru( std::vector<hypertea::RNNOp_CPU<float>* >{
      new hypertea::BidirectionalRNN_CPU<float> (
        1, 64, 32, 
        _w_ih0.data(), r_w_ih0.data(), 
        _w_hh0.data(), r_w_hh0.data(), 
        _b_ih0.data(), r_b_ih0.data(), 
        _b_hh0.data(), r_b_hh0.data(), 
        hypertea::RNN_CELL_TYPE::GRU_CELL
      )
    }
  );


  output_tensor = gru.Forward(
    input_tensor,
    std::vector<hypertea::TensorCPU<float> > {hidden_tensor}
  );


  const float* temp_data = output_tensor.immutable_data();

  pass_test(std::string("test_bi_single_gru"), hypertea::test_result::bi_single_gru_result, temp_data, float(0.001));

}



void test_uni_multi3_gru() {
    
  fake_random_number random_generator;

  auto _w_ih0 = random_generator.generate_random_vector(3*32*64);
  auto _w_hh0 = random_generator.generate_random_vector(3*32*32);
  auto _b_ih0 = random_generator.generate_random_vector(3*32);
  auto _b_hh0 = random_generator.generate_random_vector(3*32);

  auto _w_ih1 = random_generator.generate_random_vector(3*32*32);
  auto _w_hh1 = random_generator.generate_random_vector(3*32*32);
  auto _b_ih1 = random_generator.generate_random_vector(3*32);
  auto _b_hh1 = random_generator.generate_random_vector(3*32);


  auto _w_ih2 = random_generator.generate_random_vector(3*32*32);
  auto _w_hh2 = random_generator.generate_random_vector(3*32*32);
  auto _b_ih2 = random_generator.generate_random_vector(3*32);
  auto _b_hh2 = random_generator.generate_random_vector(3*32);


  auto input_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(5*64), std::vector<int>{64});
  auto hidden_tensor0 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(1*32), std::vector<int>{32});
  auto hidden_tensor1 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(1*32), std::vector<int>{32});
  auto hidden_tensor2 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(1*32), std::vector<int>{32});
  auto output_tensor = hypertea::TensorCPU<float>(32);

  hypertea::StackedRNN<float> gru( std::vector<hypertea::RNNOp_CPU<float>* >{
    new hypertea::UnidirectionalRNN_CPU<float> (
      1, 64, 32, 
      _w_ih0.data(), _w_hh0.data(), 
      _b_ih0.data(), _b_hh0.data(), 
      hypertea::RNN_CELL_TYPE::GRU_CELL
    ),

    new hypertea::UnidirectionalRNN_CPU<float> (
      1, 32, 32, 
      _w_ih1.data(), _w_hh1.data(), 
      _b_ih1.data(), _b_hh1.data(), 
      hypertea::RNN_CELL_TYPE::GRU_CELL
    ),
    new hypertea::UnidirectionalRNN_CPU<float> (
      1, 32, 32, 
      _w_ih2.data(), _w_hh2.data(), 
      _b_ih2.data(), _b_hh2.data(), 
      hypertea::RNN_CELL_TYPE::GRU_CELL
    )
    }
  );


  output_tensor = gru.Forward(
    input_tensor,
    std::vector<hypertea::TensorCPU<float> > {hidden_tensor0, hidden_tensor1, hidden_tensor2}
  );


  const float* temp_data = output_tensor.immutable_data();
  pass_test(std::string("test_uni_multi3_gru"), hypertea::test_result::uni_multi3_gru_result, temp_data, float(0.001));

}



void test_bi_multi3_gru() {
    
  fake_random_number random_generator;

  auto _w_ih0 = random_generator.generate_random_vector(3*32*64);
  auto _w_hh0 = random_generator.generate_random_vector(3*32*32);
  auto _b_ih0 = random_generator.generate_random_vector(3*32);
  auto _b_hh0 = random_generator.generate_random_vector(3*32);
  auto r_w_ih0 = random_generator.generate_random_vector(3*32*64);
  auto r_w_hh0 = random_generator.generate_random_vector(3*32*32);
  auto r_b_ih0 = random_generator.generate_random_vector(3*32);
  auto r_b_hh0 = random_generator.generate_random_vector(3*32);



  auto _w_ih1 = random_generator.generate_random_vector(3*32*64);
  auto _w_hh1 = random_generator.generate_random_vector(3*32*32);
  auto _b_ih1 = random_generator.generate_random_vector(3*32);
  auto _b_hh1 = random_generator.generate_random_vector(3*32);
  auto r_w_ih1 = random_generator.generate_random_vector(3*32*64);
  auto r_w_hh1 = random_generator.generate_random_vector(3*32*32);
  auto r_b_ih1 = random_generator.generate_random_vector(3*32);
  auto r_b_hh1 = random_generator.generate_random_vector(3*32);


  auto _w_ih2 = random_generator.generate_random_vector(3*32*64);
  auto _w_hh2 = random_generator.generate_random_vector(3*32*32);
  auto _b_ih2 = random_generator.generate_random_vector(3*32);
  auto _b_hh2 = random_generator.generate_random_vector(3*32);
  auto r_w_ih2 = random_generator.generate_random_vector(3*32*64);
  auto r_w_hh2 = random_generator.generate_random_vector(3*32*32);
  auto r_b_ih2 = random_generator.generate_random_vector(3*32);
  auto r_b_hh2 = random_generator.generate_random_vector(3*32);


  auto input_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(5*64), std::vector<int>{64});
  auto hidden_tensor0 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(2*32), std::vector<int>{32});
  auto hidden_tensor1 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(2*32), std::vector<int>{32});
  auto hidden_tensor2 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(2*32), std::vector<int>{32});
  auto output_tensor = hypertea::TensorCPU<float>(32);

  hypertea::StackedRNN<float> gru( std::vector<hypertea::RNNOp_CPU<float>* >{

      new hypertea::BidirectionalRNN_CPU<float> (
        1, 64, 32, 
        _w_ih0.data(), r_w_ih0.data(), 
        _w_hh0.data(), r_w_hh0.data(), 
        _b_ih0.data(), r_b_ih0.data(), 
        _b_hh0.data(), r_b_hh0.data(), 
        hypertea::RNN_CELL_TYPE::GRU_CELL
      ),
      new hypertea::BidirectionalRNN_CPU<float> (
        1, 64, 32, 
        _w_ih1.data(), r_w_ih1.data(), 
        _w_hh1.data(), r_w_hh1.data(), 
        _b_ih1.data(), r_b_ih1.data(), 
        _b_hh1.data(), r_b_hh1.data(), 
        hypertea::RNN_CELL_TYPE::GRU_CELL
      ),      
      new hypertea::BidirectionalRNN_CPU<float> (
        1, 64, 32, 
        _w_ih2.data(), r_w_ih2.data(), 
        _w_hh2.data(), r_w_hh2.data(), 
        _b_ih2.data(), r_b_ih2.data(), 
        _b_hh2.data(), r_b_hh2.data(), 
        hypertea::RNN_CELL_TYPE::GRU_CELL
      )
    }
  );


  output_tensor = gru.Forward(
    input_tensor,
    std::vector<hypertea::TensorCPU<float> > {hidden_tensor0, hidden_tensor1, hidden_tensor2}
  );


  const float* temp_data = output_tensor.immutable_data();
  pass_test(std::string("test_bi_multi3_gru"), hypertea::test_result::bi_multi3_gru_result, temp_data, float(0.001));

}



void test_gru() {
  test_uni_single_gru();
  test_bi_single_gru();
  test_uni_multi3_gru();
  test_bi_multi3_gru();
}







void test_uni_single_lstm() {
    
  fake_random_number random_generator;

  auto _w_ih0 = random_generator.generate_random_vector(4*32*64);
  auto _w_hh0 = random_generator.generate_random_vector(4*32*32);
  auto _b_ih0 = random_generator.generate_random_vector(4*32);
  auto _b_hh0 = random_generator.generate_random_vector(4*32);


  auto input_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(5*64), std::vector<int>{64});
  auto hidden_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(2*32), std::vector<int>{32});
  auto output_tensor = hypertea::TensorCPU<float>(32);

  hypertea::StackedRNN<float> lstm( std::vector<hypertea::RNNOp_CPU<float>* >{
      new hypertea::UnidirectionalRNN_CPU<float> (
        1, 64, 32, 
        _w_ih0.data(), _w_hh0.data(), 
        _b_ih0.data(), _b_hh0.data(), 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      )
    }
  );


  output_tensor = lstm.Forward(
    input_tensor,
    std::vector<hypertea::TensorCPU<float> > {hidden_tensor}
  );

  const float* temp_data = output_tensor.immutable_data();
  pass_test(std::string("test_uni_single_lstm"), hypertea::test_result::uni_single_lstm_result, temp_data, float(0.001));

}


void test_bi_single_lstm() {
    
  fake_random_number random_generator;

  auto _w_ih0 = random_generator.generate_random_vector(4*32*64);
  auto _w_hh0 = random_generator.generate_random_vector(4*32*32);
  auto _b_ih0 = random_generator.generate_random_vector(4*32);
  auto _b_hh0 = random_generator.generate_random_vector(4*32);

  auto r_w_ih0 = random_generator.generate_random_vector(4*32*64);
  auto r_w_hh0 = random_generator.generate_random_vector(4*32*32);
  auto r_b_ih0 = random_generator.generate_random_vector(4*32);
  auto r_b_hh0 = random_generator.generate_random_vector(4*32);

  auto input_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(5*64), std::vector<int>{64});
  auto hidden_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(4*32), std::vector<int>{32});
  auto output_tensor = hypertea::TensorCPU<float>(32);

  hypertea::StackedRNN<float> lstm( std::vector<hypertea::RNNOp_CPU<float>* >{
      new hypertea::BidirectionalRNN_CPU<float> (
        1, 64, 32, 
        _w_ih0.data(), r_w_ih0.data(), 
        _w_hh0.data(), r_w_hh0.data(), 
        _b_ih0.data(), r_b_ih0.data(), 
        _b_hh0.data(), r_b_hh0.data(), 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      )
    }
  );


  output_tensor = lstm.Forward(
    input_tensor,
    std::vector<hypertea::TensorCPU<float> > {hidden_tensor}
  );


  const float* temp_data = output_tensor.immutable_data();
  pass_test(std::string("test_bi_single_lstm"), hypertea::test_result::bi_single_lstm_result, temp_data, float(0.001));

}



void test_uni_multi3_lstm() {
    
  fake_random_number random_generator;

  auto _w_ih0 = random_generator.generate_random_vector(4*32*64);
  auto _w_hh0 = random_generator.generate_random_vector(4*32*32);
  auto _b_ih0 = random_generator.generate_random_vector(4*32);
  auto _b_hh0 = random_generator.generate_random_vector(4*32);

  auto _w_ih1 = random_generator.generate_random_vector(4*32*32);
  auto _w_hh1 = random_generator.generate_random_vector(4*32*32);
  auto _b_ih1 = random_generator.generate_random_vector(4*32);
  auto _b_hh1 = random_generator.generate_random_vector(4*32);


  auto _w_ih2 = random_generator.generate_random_vector(4*32*32);
  auto _w_hh2 = random_generator.generate_random_vector(4*32*32);
  auto _b_ih2 = random_generator.generate_random_vector(4*32);
  auto _b_hh2 = random_generator.generate_random_vector(4*32);


  auto input_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(5*64), std::vector<int>{64});
  auto hidden_tensor0 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(2*32), std::vector<int>{32});
  auto hidden_tensor1 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(2*32), std::vector<int>{32});
  auto hidden_tensor2 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(2*32), std::vector<int>{32});
  auto output_tensor = hypertea::TensorCPU<float>(32);

  hypertea::StackedRNN<float> lstm( std::vector<hypertea::RNNOp_CPU<float>* >{
      new hypertea::UnidirectionalRNN_CPU<float> (
        1, 64, 32, 
        _w_ih0.data(), _w_hh0.data(), 
        _b_ih0.data(), _b_hh0.data(), 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      ),

      new hypertea::UnidirectionalRNN_CPU<float> (
        1, 32, 32, 
        _w_ih1.data(), _w_hh1.data(), 
        _b_ih1.data(), _b_hh1.data(), 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      ),
      new hypertea::UnidirectionalRNN_CPU<float> (
        1, 32, 32, 
        _w_ih2.data(), _w_hh2.data(), 
        _b_ih2.data(), _b_hh2.data(), 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      )
    }
  );


  output_tensor = lstm.Forward(
    input_tensor,
    std::vector<hypertea::TensorCPU<float> > {hidden_tensor0, hidden_tensor1, hidden_tensor2}
  );

  const float* temp_data = output_tensor.immutable_data();
  pass_test(std::string("test_uni_multi3_lstm"), hypertea::test_result::uni_multi3_lstm_result, temp_data, float(0.001));

}




void test_bi_multi3_lstm() {
    
  fake_random_number random_generator;

  auto _w_ih0 = random_generator.generate_random_vector(4*32*64);
  auto _w_hh0 = random_generator.generate_random_vector(4*32*32);
  auto _b_ih0 = random_generator.generate_random_vector(4*32);
  auto _b_hh0 = random_generator.generate_random_vector(4*32);
  auto r_w_ih0 = random_generator.generate_random_vector(4*32*64);
  auto r_w_hh0 = random_generator.generate_random_vector(4*32*32);
  auto r_b_ih0 = random_generator.generate_random_vector(4*32);
  auto r_b_hh0 = random_generator.generate_random_vector(4*32);



  auto _w_ih1 = random_generator.generate_random_vector(4*32*64);
  auto _w_hh1 = random_generator.generate_random_vector(4*32*32);
  auto _b_ih1 = random_generator.generate_random_vector(4*32);
  auto _b_hh1 = random_generator.generate_random_vector(4*32);
  auto r_w_ih1 = random_generator.generate_random_vector(4*32*64);
  auto r_w_hh1 = random_generator.generate_random_vector(4*32*32);
  auto r_b_ih1 = random_generator.generate_random_vector(4*32);
  auto r_b_hh1 = random_generator.generate_random_vector(4*32);


  auto _w_ih2 = random_generator.generate_random_vector(4*32*64);
  auto _w_hh2 = random_generator.generate_random_vector(4*32*32);
  auto _b_ih2 = random_generator.generate_random_vector(4*32);
  auto _b_hh2 = random_generator.generate_random_vector(4*32);
  auto r_w_ih2 = random_generator.generate_random_vector(4*32*64);
  auto r_w_hh2 = random_generator.generate_random_vector(4*32*32);
  auto r_b_ih2 = random_generator.generate_random_vector(4*32);
  auto r_b_hh2 = random_generator.generate_random_vector(4*32);


  auto input_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(5*64), std::vector<int>{64});
  auto hidden_tensor0 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(4*32), std::vector<int>{32});
  auto hidden_tensor1 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(4*32), std::vector<int>{32});
  auto hidden_tensor2 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(4*32), std::vector<int>{32});
  auto output_tensor = hypertea::TensorCPU<float>(32);

  hypertea::StackedRNN<float> lstm( std::vector<hypertea::RNNOp_CPU<float>* >{
      new hypertea::BidirectionalRNN_CPU<float> (
        1, 64, 32, 
        _w_ih0.data(), r_w_ih0.data(), 
        _w_hh0.data(), r_w_hh0.data(), 
        _b_ih0.data(), r_b_ih0.data(), 
        _b_hh0.data(), r_b_hh0.data(), 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      ),
      new hypertea::BidirectionalRNN_CPU<float> (
        1, 64, 32, 
        _w_ih1.data(), r_w_ih1.data(), 
        _w_hh1.data(), r_w_hh1.data(), 
        _b_ih1.data(), r_b_ih1.data(), 
        _b_hh1.data(), r_b_hh1.data(), 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      ),      
      new hypertea::BidirectionalRNN_CPU<float> (
        1, 64, 32, 
        _w_ih2.data(), r_w_ih2.data(), 
        _w_hh2.data(), r_w_hh2.data(), 
        _b_ih2.data(), r_b_ih2.data(), 
        _b_hh2.data(), r_b_hh2.data(), 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      )
    }
  );


  output_tensor = lstm.Forward(
    input_tensor,
    std::vector<hypertea::TensorCPU<float> > {hidden_tensor0, hidden_tensor1, hidden_tensor2}
  );



  const float* temp_data = output_tensor.immutable_data();

  pass_test(std::string("test_bi_multi3_lstm"), hypertea::test_result::bi_multi3_lstm_result, temp_data, float(0.001));

}

void test_lstm() {
  test_uni_single_lstm();
  test_bi_single_lstm();
  test_uni_multi3_lstm();
  test_bi_multi3_lstm();
}



int main(int argc, char** argv) {

  test_gru();
  test_lstm();
  exit(0);

  fake_random_number random_generator;

  auto _w_ih0 = random_generator.generate_random_vector(4*32*64);
  auto _w_hh0 = random_generator.generate_random_vector(4*32*32);
  auto _b_ih0 = random_generator.generate_random_vector(4*32);
  auto _b_hh0 = random_generator.generate_random_vector(4*32);

  auto r_w_ih0 = random_generator.generate_random_vector(4*32*64);
  auto r_w_hh0 = random_generator.generate_random_vector(4*32*32);
  auto r_b_ih0 = random_generator.generate_random_vector(4*32);
  auto r_b_hh0 = random_generator.generate_random_vector(4*32);


  auto input_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(5*64), std::vector<int>{64});
  auto hidden_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(4*32), std::vector<int>{32});
  auto output_tensor = hypertea::TensorCPU<float>(32);

  hypertea::StackedRNN<float> lstm( std::vector<hypertea::RNNOp_CPU<float>* >{
      new hypertea::BidirectionalRNN_CPU<float> (1, 64, 32, _w_ih0.data(), r_w_ih0.data(), _w_hh0.data(), r_w_hh0.data(), _b_ih0.data(), r_b_ih0.data(), _b_hh0.data(), r_b_hh0.data(), hypertea::RNN_CELL_TYPE::LSTM_CELL)
    }
  );


  output_tensor = lstm.Forward(
    input_tensor,
    std::vector<hypertea::TensorCPU<float> > {hidden_tensor}
  );


  const float* temp_data = output_tensor.immutable_data();


  for (int i = 0; i < output_tensor.count(); ++i) {
    
    if(i % 32 == 0) {
      std::cout << " " << std::endl << std::endl;
    }

    std::cout << temp_data[i] << " ";
    
  }

    std::cout << "\nThe total number of output is "<< output_tensor.count() << std::endl;


}
