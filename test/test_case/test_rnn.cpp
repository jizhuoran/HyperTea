#include <algorithm>
#include <vector>

#include "gtest/gtest.h"


#include "hypertea/common.hpp"

#include "hypertea/operators/rnn_op.hpp"
#include "test_result/rnn_result.hpp"

// #include "caffe/blob.hpp"
// #include "caffe/common.hpp"
// #include "caffe/filler.hpp"
// #include "caffe/layers/bias_layer.hpp"

#include "test_hypertea_util.hpp"
// #include "caffe/test/test_gradient_check_util.hpp"

namespace hypertea {


template <typename TypeParam>
class RNNTestCPU : public ::testing::Test {
 public:
  typedef typename TypeParam::Dtype Dtype;
 protected:
  RNNTestCPU() {
    // hypertea::OpenCLHandler::Get().build_opencl_math_code(false);
  }
  virtual ~RNNTestCPU() {}
};

template <typename TypeParam>
class RNNTestGPU : public ::testing::Test {
 public:
  typedef typename TypeParam::Dtype Dtype;
 protected:
  RNNTestGPU() {
    hypertea::OpenCLHandler::Get().build_opencl_math_code(false);
  }
  virtual ~RNNTestGPU() {}
};


TYPED_TEST_CASE(RNNTestCPU, TestDtypesCPU);
TYPED_TEST_CASE(RNNTestGPU, TestDtypesGPU);



TYPED_TEST(RNNTestCPU, test_uni_single_gru_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto _w_ih0 = random_generator.generate_random_vector(3*32*64);
  auto _w_hh0 = random_generator.generate_random_vector(3*32*32);
  auto _b_ih0 = random_generator.generate_random_vector(3*32);
  auto _b_hh0 = random_generator.generate_random_vector(3*32);


  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(5*64));
  auto hidden_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(1*32));
  auto output_tensor = hypertea::TensorCPU<Dtype>(32);

  hypertea::StackedRNN_CPU<Dtype> gru( std::vector<hypertea::RNNOp_CPU<Dtype>* >{
      new hypertea::UnidirectionalRNN_CPU<Dtype> (
        64, 32, 
        _w_ih0.data(), _w_hh0.data(), 
        _b_ih0.data(), _b_hh0.data(), 
        hypertea::RNN_CELL_TYPE::GRU_CELL
      )
    }
  );

  output_tensor = gru.Forward(
    input_tensor,
    std::vector<hypertea::TensorCPU<Dtype> > {hidden_tensor}
  );

  const Dtype* output_data = output_tensor.immutable_data();

  for (int i = 0; i < test_result::uni_single_gru_result.size(); ++i) {
    EXPECT_NEAR(output_data[i], test_result::uni_single_gru_result[i], 1e-3);
  }
}


TYPED_TEST(RNNTestCPU, test_bi_single_gru_CPU) {
  
  typedef typename TypeParam::Dtype Dtype;
    
  fake_random_number random_generator;

  auto _w_ih0 = random_generator.generate_random_vector(3*32*64);
  auto _w_hh0 = random_generator.generate_random_vector(3*32*32);
  auto _b_ih0 = random_generator.generate_random_vector(3*32);
  auto _b_hh0 = random_generator.generate_random_vector(3*32);

  auto r_w_ih0 = random_generator.generate_random_vector(3*32*64);
  auto r_w_hh0 = random_generator.generate_random_vector(3*32*32);
  auto r_b_ih0 = random_generator.generate_random_vector(3*32);
  auto r_b_hh0 = random_generator.generate_random_vector(3*32);

  auto input_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(5*64));
  auto hidden_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(2*32));
  auto output_tensor = hypertea::TensorCPU<float>(32);

  hypertea::StackedRNN_CPU<float> gru( std::vector<hypertea::RNNOp_CPU<float>* >{
      new hypertea::BidirectionalRNN_CPU<float> (
        64, 32, 
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


  const Dtype* output_data = output_tensor.immutable_data();

  for (int i = 0; i < test_result::bi_single_gru_result.size(); ++i) {
    EXPECT_NEAR(output_data[i], test_result::bi_single_gru_result[i], 1e-3);
  }
}


TYPED_TEST(RNNTestCPU, test_uni_multi3_gru_CPU) {
  
  typedef typename TypeParam::Dtype Dtype;
    
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


  auto input_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(5*64));
  auto hidden_tensor0 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(1*32));
  auto hidden_tensor1 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(1*32));
  auto hidden_tensor2 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(1*32));
  auto output_tensor = hypertea::TensorCPU<float>(32);

  hypertea::StackedRNN_CPU<float> gru( std::vector<hypertea::RNNOp_CPU<float>* >{
    new hypertea::UnidirectionalRNN_CPU<float> (
      64, 32, 
      _w_ih0.data(), _w_hh0.data(), 
      _b_ih0.data(), _b_hh0.data(), 
      hypertea::RNN_CELL_TYPE::GRU_CELL
    ),

    new hypertea::UnidirectionalRNN_CPU<float> (
      32, 32, 
      _w_ih1.data(), _w_hh1.data(), 
      _b_ih1.data(), _b_hh1.data(), 
      hypertea::RNN_CELL_TYPE::GRU_CELL
    ),
    new hypertea::UnidirectionalRNN_CPU<float> (
      32, 32, 
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


  const Dtype* output_data = output_tensor.immutable_data();
  for (int i = 0; i < test_result::uni_multi3_gru_result.size(); ++i) {
    EXPECT_NEAR(output_data[i], test_result::uni_multi3_gru_result[i], 1e-3);
  }

}


TYPED_TEST(RNNTestCPU, test_bi_multi3_gru_CPU) {

  typedef typename TypeParam::Dtype Dtype;

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


  auto input_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(5*64));
  auto hidden_tensor0 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(2*32));
  auto hidden_tensor1 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(2*32));
  auto hidden_tensor2 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(2*32));
  auto output_tensor = hypertea::TensorCPU<float>(32);

  hypertea::StackedRNN_CPU<float> gru( std::vector<hypertea::RNNOp_CPU<float>* >{

      new hypertea::BidirectionalRNN_CPU<float> (
        64, 32, 
        _w_ih0.data(), r_w_ih0.data(), 
        _w_hh0.data(), r_w_hh0.data(), 
        _b_ih0.data(), r_b_ih0.data(), 
        _b_hh0.data(), r_b_hh0.data(), 
        hypertea::RNN_CELL_TYPE::GRU_CELL
      ),
      new hypertea::BidirectionalRNN_CPU<float> (
        64, 32, 
        _w_ih1.data(), r_w_ih1.data(), 
        _w_hh1.data(), r_w_hh1.data(), 
        _b_ih1.data(), r_b_ih1.data(), 
        _b_hh1.data(), r_b_hh1.data(), 
        hypertea::RNN_CELL_TYPE::GRU_CELL
      ),      
      new hypertea::BidirectionalRNN_CPU<float> (
        64, 32, 
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

  const Dtype* output_data = output_tensor.immutable_data();
  for (int i = 0; i < test_result::bi_multi3_gru_result.size(); ++i) {
    EXPECT_NEAR(output_data[i], test_result::bi_multi3_gru_result[i], 1e-3);
  }

}





TYPED_TEST(RNNTestCPU, test_uni_single_lstm_CPU) {

  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto _w_ih0 = random_generator.generate_random_vector(4*32*64);
  auto _w_hh0 = random_generator.generate_random_vector(4*32*32);
  auto _b_ih0 = random_generator.generate_random_vector(4*32);
  auto _b_hh0 = random_generator.generate_random_vector(4*32);


  auto input_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(5*64));
  auto hidden_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(2*32));
  auto output_tensor = hypertea::TensorCPU<float>(32);

  hypertea::StackedRNN_CPU<float> lstm( std::vector<hypertea::RNNOp_CPU<float>* >{
      new hypertea::UnidirectionalRNN_CPU<float> (
        64, 32, 
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

  const Dtype* output_data = output_tensor.immutable_data();
  for (int i = 0; i < test_result::uni_single_lstm_result.size(); ++i) {
    EXPECT_NEAR(output_data[i], test_result::uni_single_lstm_result[i], 1e-3);
  }

}


TYPED_TEST(RNNTestCPU, test_bi_single_lstm_CPU) {

  typedef typename TypeParam::Dtype Dtype;

  fake_random_number random_generator;

  auto _w_ih0 = random_generator.generate_random_vector(4*32*64);
  auto _w_hh0 = random_generator.generate_random_vector(4*32*32);
  auto _b_ih0 = random_generator.generate_random_vector(4*32);
  auto _b_hh0 = random_generator.generate_random_vector(4*32);

  auto r_w_ih0 = random_generator.generate_random_vector(4*32*64);
  auto r_w_hh0 = random_generator.generate_random_vector(4*32*32);
  auto r_b_ih0 = random_generator.generate_random_vector(4*32);
  auto r_b_hh0 = random_generator.generate_random_vector(4*32);

  auto input_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(5*64));
  auto hidden_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(4*32));
  auto output_tensor = hypertea::TensorCPU<float>(32);

  hypertea::StackedRNN_CPU<float> lstm( std::vector<hypertea::RNNOp_CPU<float>* >{
      new hypertea::BidirectionalRNN_CPU<float> (
        64, 32, 
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


  const Dtype* output_data = output_tensor.immutable_data();
  for (int i = 0; i < test_result::bi_single_lstm_result.size(); ++i) {
    EXPECT_NEAR(output_data[i], test_result::bi_single_lstm_result[i], 1e-3);
  }

}



TYPED_TEST(RNNTestCPU, test_uni_multi3_lstm_CPU) {

  typedef typename TypeParam::Dtype Dtype;

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


  auto input_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(5*64));
  auto hidden_tensor0 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(2*32));
  auto hidden_tensor1 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(2*32));
  auto hidden_tensor2 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(2*32));
  auto output_tensor = hypertea::TensorCPU<float>(32);

  hypertea::StackedRNN_CPU<float> lstm( std::vector<hypertea::RNNOp_CPU<float>* >{
      new hypertea::UnidirectionalRNN_CPU<float> (
        64, 32, 
        _w_ih0.data(), _w_hh0.data(), 
        _b_ih0.data(), _b_hh0.data(), 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      ),

      new hypertea::UnidirectionalRNN_CPU<float> (
        32, 32, 
        _w_ih1.data(), _w_hh1.data(), 
        _b_ih1.data(), _b_hh1.data(), 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      ),
      new hypertea::UnidirectionalRNN_CPU<float> (
        32, 32, 
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

  const Dtype* output_data = output_tensor.immutable_data();
  for (int i = 0; i < test_result::uni_multi3_lstm_result.size(); ++i) {
    EXPECT_NEAR(output_data[i], test_result::uni_multi3_lstm_result[i], 1e-3);
  }

}




TYPED_TEST(RNNTestCPU, test_bi_multi3_lstm_CPU) {

  typedef typename TypeParam::Dtype Dtype;

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


  auto input_tensor = hypertea::TensorCPU<float>(random_generator.generate_random_vector(5*64));
  auto hidden_tensor0 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(4*32));
  auto hidden_tensor1 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(4*32));
  auto hidden_tensor2 = hypertea::TensorCPU<float>(random_generator.generate_random_vector(4*32));
  auto output_tensor = hypertea::TensorCPU<float>(32);

  hypertea::StackedRNN_CPU<float> lstm( std::vector<hypertea::RNNOp_CPU<float>* >{
      new hypertea::BidirectionalRNN_CPU<float> (
        64, 32, 
        _w_ih0.data(), r_w_ih0.data(), 
        _w_hh0.data(), r_w_hh0.data(), 
        _b_ih0.data(), r_b_ih0.data(), 
        _b_hh0.data(), r_b_hh0.data(), 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      ),
      new hypertea::BidirectionalRNN_CPU<float> (
        64, 32, 
        _w_ih1.data(), r_w_ih1.data(), 
        _w_hh1.data(), r_w_hh1.data(), 
        _b_ih1.data(), r_b_ih1.data(), 
        _b_hh1.data(), r_b_hh1.data(), 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      ),      
      new hypertea::BidirectionalRNN_CPU<float> (
        64, 32, 
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

  const Dtype* output_data = output_tensor.immutable_data();
  for (int i = 0; i < test_result::bi_multi3_lstm_result.size(); ++i) {
    EXPECT_NEAR(output_data[i], test_result::bi_multi3_lstm_result[i], 1e-3);
  }

}



TYPED_TEST(RNNTestGPU, test_uni_single_gru_GPU) {
    
  typedef typename TypeParam::Dtype Dtype;

  fake_random_number random_generator;

  auto _w_ih0_vec = random_generator.generate_random_vector(3*32*64);
  auto _w_hh0_vec = random_generator.generate_random_vector(3*32*32);
  auto _b_ih0_vec = random_generator.generate_random_vector(3*32);
  auto _b_hh0_vec = random_generator.generate_random_vector(3*32);
  auto _w_ih0 = hypertea::TensorGPU<float>(_w_ih0_vec); 
  auto _w_hh0 = hypertea::TensorGPU<float>(_w_hh0_vec); 
  auto _b_ih0 = hypertea::TensorGPU<float>(_b_ih0_vec); 
  auto _b_hh0 = hypertea::TensorGPU<float>(_b_hh0_vec); 


  auto input_tensor = hypertea::TensorGPU<float>(random_generator.generate_random_vector(5*64));
  auto hidden_tensor = hypertea::TensorGPU<float>(random_generator.generate_random_vector(32));
  auto output_tensor = hypertea::TensorGPU<float>(32);

  hypertea::StackedRNN_GPU<float> gru( std::vector<hypertea::RNNOp_GPU<float>* >{
      new hypertea::UnidirectionalRNN_GPU<float> (
        64, 32, 
        _w_ih0.mutable_data(), _w_hh0.mutable_data(), 
        _b_ih0.mutable_data(), _b_hh0.mutable_data(), 
        hypertea::RNN_CELL_TYPE::GRU_CELL
      )
    }
  );


  output_tensor = gru.Forward(
    input_tensor,
    std::vector<hypertea::TensorGPU<float> > {hidden_tensor}
  );

  const float* output_data = output_tensor.debug_cpu_data();
  for (int i = 0; i < test_result::uni_single_gru_result.size(); ++i) {
    EXPECT_NEAR(output_data[i], test_result::uni_single_gru_result[i], 1e-3);
  }

}


TYPED_TEST(RNNTestGPU, test_bi_single_gru_GPU) {
    
  typedef typename TypeParam::Dtype Dtype;

  fake_random_number random_generator;

  auto _w_ih0_vec = random_generator.generate_random_vector(3*32*64);
  auto _w_hh0_vec = random_generator.generate_random_vector(3*32*32);
  auto _b_ih0_vec = random_generator.generate_random_vector(3*32);
  auto _b_hh0_vec = random_generator.generate_random_vector(3*32);
  auto _w_ih0 = hypertea::TensorGPU<float>(_w_ih0_vec); 
  auto _w_hh0 = hypertea::TensorGPU<float>(_w_hh0_vec); 
  auto _b_ih0 = hypertea::TensorGPU<float>(_b_ih0_vec); 
  auto _b_hh0 = hypertea::TensorGPU<float>(_b_hh0_vec); 


  auto r_w_ih0_vec = random_generator.generate_random_vector(3*32*64);
  auto r_w_hh0_vec = random_generator.generate_random_vector(3*32*32);
  auto r_b_ih0_vec = random_generator.generate_random_vector(3*32);
  auto r_b_hh0_vec = random_generator.generate_random_vector(3*32);
  auto r_w_ih0 = hypertea::TensorGPU<float>(r_w_ih0_vec); 
  auto r_w_hh0 = hypertea::TensorGPU<float>(r_w_hh0_vec); 
  auto r_b_ih0 = hypertea::TensorGPU<float>(r_b_ih0_vec); 
  auto r_b_hh0 = hypertea::TensorGPU<float>(r_b_hh0_vec); 


  auto input_tensor = hypertea::TensorGPU<float>(random_generator.generate_random_vector(5*64));
  auto hidden_tensor = hypertea::TensorGPU<float>(random_generator.generate_random_vector(2*32));
  auto output_tensor = hypertea::TensorGPU<float>(32);

  hypertea::StackedRNN_GPU<float> gru( std::vector<hypertea::RNNOp_GPU<float>* >{
      new hypertea::BidirectionalRNN_GPU<float> (
        64, 32, 
        _w_ih0.mutable_data(), r_w_ih0.mutable_data(), 
        _w_hh0.mutable_data(), r_w_hh0.mutable_data(), 
        _b_ih0.mutable_data(), r_b_ih0.mutable_data(), 
        _b_hh0.mutable_data(), r_b_hh0.mutable_data(), 
        hypertea::RNN_CELL_TYPE::GRU_CELL
      )
    }
  );


  output_tensor = gru.Forward(
    input_tensor,
    std::vector<hypertea::TensorGPU<float> > {hidden_tensor}
  );


  const float* output_data = output_tensor.debug_cpu_data();
  for (int i = 0; i < test_result::bi_single_gru_result.size(); ++i) {
    EXPECT_NEAR(output_data[i], test_result::bi_single_gru_result[i], 1e-3);
  }

}



TYPED_TEST(RNNTestGPU, test_uni_multi3_gru_GPU) {
    
  typedef typename TypeParam::Dtype Dtype;

  fake_random_number random_generator;

  auto _w_ih0_vec = random_generator.generate_random_vector(3*32*64);
  auto _w_hh0_vec = random_generator.generate_random_vector(3*32*32);
  auto _b_ih0_vec = random_generator.generate_random_vector(3*32);
  auto _b_hh0_vec = random_generator.generate_random_vector(3*32);
  auto _w_ih0 = hypertea::TensorGPU<float>(_w_ih0_vec); 
  auto _w_hh0 = hypertea::TensorGPU<float>(_w_hh0_vec); 
  auto _b_ih0 = hypertea::TensorGPU<float>(_b_ih0_vec); 
  auto _b_hh0 = hypertea::TensorGPU<float>(_b_hh0_vec); 

  auto _w_ih1_vec = random_generator.generate_random_vector(3*32*32);
  auto _w_hh1_vec = random_generator.generate_random_vector(3*32*32);
  auto _b_ih1_vec = random_generator.generate_random_vector(3*32);
  auto _b_hh1_vec = random_generator.generate_random_vector(3*32);
  auto _w_ih1 = hypertea::TensorGPU<float>(_w_ih1_vec); 
  auto _w_hh1 = hypertea::TensorGPU<float>(_w_hh1_vec); 
  auto _b_ih1 = hypertea::TensorGPU<float>(_b_ih1_vec); 
  auto _b_hh1 = hypertea::TensorGPU<float>(_b_hh1_vec); 

  auto _w_ih2_vec = random_generator.generate_random_vector(3*32*32);
  auto _w_hh2_vec = random_generator.generate_random_vector(3*32*32);
  auto _b_ih2_vec = random_generator.generate_random_vector(3*32);
  auto _b_hh2_vec = random_generator.generate_random_vector(3*32);
  auto _w_ih2 = hypertea::TensorGPU<float>(_w_ih2_vec); 
  auto _w_hh2 = hypertea::TensorGPU<float>(_w_hh2_vec); 
  auto _b_ih2 = hypertea::TensorGPU<float>(_b_ih2_vec); 
  auto _b_hh2 = hypertea::TensorGPU<float>(_b_hh2_vec); 

  auto input_tensor = hypertea::TensorGPU<float>(random_generator.generate_random_vector(5*64));
  auto hidden_tensor0 = hypertea::TensorGPU<float>(random_generator.generate_random_vector(32));
  auto hidden_tensor1 = hypertea::TensorGPU<float>(random_generator.generate_random_vector(32));
  auto hidden_tensor2 = hypertea::TensorGPU<float>(random_generator.generate_random_vector(32));
  auto output_tensor = hypertea::TensorGPU<float>(32);

  hypertea::StackedRNN_GPU<float> gru( std::vector<hypertea::RNNOp_GPU<float>* >{
      new hypertea::UnidirectionalRNN_GPU<float> (
        64, 32, 
        _w_ih0.mutable_data(), _w_hh0.mutable_data(), 
        _b_ih0.mutable_data(), _b_hh0.mutable_data(), 
        hypertea::RNN_CELL_TYPE::GRU_CELL
      ),

      new hypertea::UnidirectionalRNN_GPU<float> (
        32, 32, 
        _w_ih1.mutable_data(), _w_hh1.mutable_data(), 
        _b_ih1.mutable_data(), _b_hh1.mutable_data(), 
        hypertea::RNN_CELL_TYPE::GRU_CELL
      ),
      new hypertea::UnidirectionalRNN_GPU<float> (
        32, 32, 
        _w_ih2.mutable_data(), _w_hh2.mutable_data(), 
        _b_ih2.mutable_data(), _b_hh2.mutable_data(), 
        hypertea::RNN_CELL_TYPE::GRU_CELL
      )
    }
  );


  output_tensor = gru.Forward(
    input_tensor,
    std::vector<hypertea::TensorGPU<float> > {hidden_tensor0, hidden_tensor1, hidden_tensor2}
  );

  const float* output_data = output_tensor.debug_cpu_data();
  for (int i = 0; i < test_result::uni_multi3_gru_result.size(); ++i) {
    EXPECT_NEAR(output_data[i], test_result::uni_multi3_gru_result[i], 1e-3);
  }

}




TYPED_TEST(RNNTestGPU, test_bi_multi3_gru_GPU) {
    
  typedef typename TypeParam::Dtype Dtype;

  fake_random_number random_generator;

  auto _w_ih0_vec = random_generator.generate_random_vector(3*32*64);
  auto _w_hh0_vec = random_generator.generate_random_vector(3*32*32);
  auto _b_ih0_vec = random_generator.generate_random_vector(3*32);
  auto _b_hh0_vec = random_generator.generate_random_vector(3*32);
  auto r_w_ih0_vec = random_generator.generate_random_vector(3*32*64);
  auto r_w_hh0_vec = random_generator.generate_random_vector(3*32*32);
  auto r_b_ih0_vec = random_generator.generate_random_vector(3*32);
  auto r_b_hh0_vec = random_generator.generate_random_vector(3*32);
  auto _w_ih0 = hypertea::TensorGPU<float>(_w_ih0_vec); 
  auto _w_hh0 = hypertea::TensorGPU<float>(_w_hh0_vec); 
  auto _b_ih0 = hypertea::TensorGPU<float>(_b_ih0_vec); 
  auto _b_hh0 = hypertea::TensorGPU<float>(_b_hh0_vec);
  auto r_w_ih0 = hypertea::TensorGPU<float>(r_w_ih0_vec); 
  auto r_w_hh0 = hypertea::TensorGPU<float>(r_w_hh0_vec); 
  auto r_b_ih0 = hypertea::TensorGPU<float>(r_b_ih0_vec); 
  auto r_b_hh0 = hypertea::TensorGPU<float>(r_b_hh0_vec);


  auto _w_ih1_vec = random_generator.generate_random_vector(3*32*64);
  auto _w_hh1_vec = random_generator.generate_random_vector(3*32*32);
  auto _b_ih1_vec = random_generator.generate_random_vector(3*32);
  auto _b_hh1_vec = random_generator.generate_random_vector(3*32);
  auto r_w_ih1_vec = random_generator.generate_random_vector(3*32*64);
  auto r_w_hh1_vec = random_generator.generate_random_vector(3*32*32);
  auto r_b_ih1_vec = random_generator.generate_random_vector(3*32);
  auto r_b_hh1_vec = random_generator.generate_random_vector(3*32);
  auto _w_ih1 = hypertea::TensorGPU<float>(_w_ih1_vec); 
  auto _w_hh1 = hypertea::TensorGPU<float>(_w_hh1_vec); 
  auto _b_ih1 = hypertea::TensorGPU<float>(_b_ih1_vec); 
  auto _b_hh1 = hypertea::TensorGPU<float>(_b_hh1_vec);
  auto r_w_ih1 = hypertea::TensorGPU<float>(r_w_ih1_vec); 
  auto r_w_hh1 = hypertea::TensorGPU<float>(r_w_hh1_vec); 
  auto r_b_ih1 = hypertea::TensorGPU<float>(r_b_ih1_vec); 
  auto r_b_hh1 = hypertea::TensorGPU<float>(r_b_hh1_vec);

  auto _w_ih2_vec = random_generator.generate_random_vector(3*32*64);
  auto _w_hh2_vec = random_generator.generate_random_vector(3*32*32);
  auto _b_ih2_vec = random_generator.generate_random_vector(3*32);
  auto _b_hh2_vec = random_generator.generate_random_vector(3*32);
  auto r_w_ih2_vec = random_generator.generate_random_vector(3*32*64);
  auto r_w_hh2_vec = random_generator.generate_random_vector(3*32*32);
  auto r_b_ih2_vec = random_generator.generate_random_vector(3*32);
  auto r_b_hh2_vec = random_generator.generate_random_vector(3*32);
  auto _w_ih2 = hypertea::TensorGPU<float>(_w_ih2_vec); 
  auto _w_hh2 = hypertea::TensorGPU<float>(_w_hh2_vec); 
  auto _b_ih2 = hypertea::TensorGPU<float>(_b_ih2_vec); 
  auto _b_hh2 = hypertea::TensorGPU<float>(_b_hh2_vec);
  auto r_w_ih2 = hypertea::TensorGPU<float>(r_w_ih2_vec); 
  auto r_w_hh2 = hypertea::TensorGPU<float>(r_w_hh2_vec); 
  auto r_b_ih2 = hypertea::TensorGPU<float>(r_b_ih2_vec); 
  auto r_b_hh2 = hypertea::TensorGPU<float>(r_b_hh2_vec);

  auto input_tensor = hypertea::TensorGPU<float>(random_generator.generate_random_vector(5*64));
  auto hidden_tensor0 = hypertea::TensorGPU<float>(random_generator.generate_random_vector(2*32));
  auto hidden_tensor1 = hypertea::TensorGPU<float>(random_generator.generate_random_vector(2*32));
  auto hidden_tensor2 = hypertea::TensorGPU<float>(random_generator.generate_random_vector(2*32));
  auto output_tensor = hypertea::TensorGPU<float>(32);

  hypertea::StackedRNN_GPU<float> gru( std::vector<hypertea::RNNOp_GPU<float>* >{
      new hypertea::BidirectionalRNN_GPU<float> (
        64, 32, 
        _w_ih0.mutable_data(), r_w_ih0.mutable_data(), 
        _w_hh0.mutable_data(), r_w_hh0.mutable_data(), 
        _b_ih0.mutable_data(), r_b_ih0.mutable_data(), 
        _b_hh0.mutable_data(), r_b_hh0.mutable_data(), 
        hypertea::RNN_CELL_TYPE::GRU_CELL
      ),
      new hypertea::BidirectionalRNN_GPU<float> (
        64, 32, 
        _w_ih1.mutable_data(), r_w_ih1.mutable_data(), 
        _w_hh1.mutable_data(), r_w_hh1.mutable_data(), 
        _b_ih1.mutable_data(), r_b_ih1.mutable_data(), 
        _b_hh1.mutable_data(), r_b_hh1.mutable_data(), 
        hypertea::RNN_CELL_TYPE::GRU_CELL
      ),      
      new hypertea::BidirectionalRNN_GPU<float> (
        64, 32, 
        _w_ih2.mutable_data(), r_w_ih2.mutable_data(), 
        _w_hh2.mutable_data(), r_w_hh2.mutable_data(), 
        _b_ih2.mutable_data(), r_b_ih2.mutable_data(), 
        _b_hh2.mutable_data(), r_b_hh2.mutable_data(), 
        hypertea::RNN_CELL_TYPE::GRU_CELL
      )
    }
  );


  output_tensor = gru.Forward(
    input_tensor,
    std::vector<hypertea::TensorGPU<float> > {hidden_tensor0, hidden_tensor1, hidden_tensor2}
  );



  const float* output_data = output_tensor.debug_cpu_data();

  for (int i = 0; i < test_result::bi_multi3_gru_result.size(); ++i) {
    EXPECT_NEAR(output_data[i], test_result::bi_multi3_gru_result[i], 1e-3);
  }

}




TYPED_TEST(RNNTestGPU, test_uni_single_lstm_GPU) {
    
  typedef typename TypeParam::Dtype Dtype;

  fake_random_number random_generator;

  auto _w_ih0_vec = random_generator.generate_random_vector(4*32*64);
  auto _w_hh0_vec = random_generator.generate_random_vector(4*32*32);
  auto _b_ih0_vec = random_generator.generate_random_vector(4*32);
  auto _b_hh0_vec = random_generator.generate_random_vector(4*32);
  auto _w_ih0 = hypertea::TensorGPU<float>(_w_ih0_vec); 
  auto _w_hh0 = hypertea::TensorGPU<float>(_w_hh0_vec); 
  auto _b_ih0 = hypertea::TensorGPU<float>(_b_ih0_vec); 
  auto _b_hh0 = hypertea::TensorGPU<float>(_b_hh0_vec); 


  auto input_tensor = hypertea::TensorGPU<float>(random_generator.generate_random_vector(5*64));
  auto hidden_tensor = hypertea::TensorGPU<float>(random_generator.generate_random_vector(2*32));
  auto output_tensor = hypertea::TensorGPU<float>(32);

  hypertea::StackedRNN_GPU<float> lstm( std::vector<hypertea::RNNOp_GPU<float>* >{
      new hypertea::UnidirectionalRNN_GPU<float> (
        64, 32, 
        _w_ih0.mutable_data(), _w_hh0.mutable_data(), 
        _b_ih0.mutable_data(), _b_hh0.mutable_data(), 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      )
    }
  );


  output_tensor = lstm.Forward(
    input_tensor,
    std::vector<hypertea::TensorGPU<float> > {hidden_tensor}
  );

  const float* output_data = output_tensor.debug_cpu_data();
  for (int i = 0; i < test_result::uni_single_lstm_result.size(); ++i) {
    EXPECT_NEAR(output_data[i], test_result::uni_single_lstm_result[i], 1e-3);
  }

}


TYPED_TEST(RNNTestGPU, test_bi_single_lstm_GPU) {
    
  typedef typename TypeParam::Dtype Dtype;

  fake_random_number random_generator;

  auto _w_ih0_vec = random_generator.generate_random_vector(4*32*64);
  auto _w_hh0_vec = random_generator.generate_random_vector(4*32*32);
  auto _b_ih0_vec = random_generator.generate_random_vector(4*32);
  auto _b_hh0_vec = random_generator.generate_random_vector(4*32);
  auto _w_ih0 = hypertea::TensorGPU<float>(_w_ih0_vec); 
  auto _w_hh0 = hypertea::TensorGPU<float>(_w_hh0_vec); 
  auto _b_ih0 = hypertea::TensorGPU<float>(_b_ih0_vec); 
  auto _b_hh0 = hypertea::TensorGPU<float>(_b_hh0_vec); 


  auto r_w_ih0_vec = random_generator.generate_random_vector(4*32*64);
  auto r_w_hh0_vec = random_generator.generate_random_vector(4*32*32);
  auto r_b_ih0_vec = random_generator.generate_random_vector(4*32);
  auto r_b_hh0_vec = random_generator.generate_random_vector(4*32);
  auto r_w_ih0 = hypertea::TensorGPU<float>(r_w_ih0_vec); 
  auto r_w_hh0 = hypertea::TensorGPU<float>(r_w_hh0_vec); 
  auto r_b_ih0 = hypertea::TensorGPU<float>(r_b_ih0_vec); 
  auto r_b_hh0 = hypertea::TensorGPU<float>(r_b_hh0_vec); 


  auto input_tensor = hypertea::TensorGPU<float>(random_generator.generate_random_vector(5*64));
  auto hidden_tensor = hypertea::TensorGPU<float>(random_generator.generate_random_vector(4*32));
  auto output_tensor = hypertea::TensorGPU<float>(32);

  hypertea::StackedRNN_GPU<float> lstm( std::vector<hypertea::RNNOp_GPU<float>* >{
      new hypertea::BidirectionalRNN_GPU<float> (
        64, 32, 
        _w_ih0.mutable_data(), r_w_ih0.mutable_data(), 
        _w_hh0.mutable_data(), r_w_hh0.mutable_data(), 
        _b_ih0.mutable_data(), r_b_ih0.mutable_data(), 
        _b_hh0.mutable_data(), r_b_hh0.mutable_data(), 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      )
    }
  );


  output_tensor = lstm.Forward(
    input_tensor,
    std::vector<hypertea::TensorGPU<float> > {hidden_tensor}
  );


  const float* output_data = output_tensor.debug_cpu_data();
  for (int i = 0; i < test_result::bi_single_lstm_result.size(); ++i) {
    EXPECT_NEAR(output_data[i], test_result::bi_single_lstm_result[i], 1e-3);
  }

}



TYPED_TEST(RNNTestGPU, test_uni_multi3_lstm_GPU) {
    
  typedef typename TypeParam::Dtype Dtype;

  fake_random_number random_generator;

  auto _w_ih0_vec = random_generator.generate_random_vector(4*32*64);
  auto _w_hh0_vec = random_generator.generate_random_vector(4*32*32);
  auto _b_ih0_vec = random_generator.generate_random_vector(4*32);
  auto _b_hh0_vec = random_generator.generate_random_vector(4*32);
  auto _w_ih0 = hypertea::TensorGPU<float>(_w_ih0_vec); 
  auto _w_hh0 = hypertea::TensorGPU<float>(_w_hh0_vec); 
  auto _b_ih0 = hypertea::TensorGPU<float>(_b_ih0_vec); 
  auto _b_hh0 = hypertea::TensorGPU<float>(_b_hh0_vec); 

  auto _w_ih1_vec = random_generator.generate_random_vector(4*32*32);
  auto _w_hh1_vec = random_generator.generate_random_vector(4*32*32);
  auto _b_ih1_vec = random_generator.generate_random_vector(4*32);
  auto _b_hh1_vec = random_generator.generate_random_vector(4*32);
  auto _w_ih1 = hypertea::TensorGPU<float>(_w_ih1_vec); 
  auto _w_hh1 = hypertea::TensorGPU<float>(_w_hh1_vec); 
  auto _b_ih1 = hypertea::TensorGPU<float>(_b_ih1_vec); 
  auto _b_hh1 = hypertea::TensorGPU<float>(_b_hh1_vec); 

  auto _w_ih2_vec = random_generator.generate_random_vector(4*32*32);
  auto _w_hh2_vec = random_generator.generate_random_vector(4*32*32);
  auto _b_ih2_vec = random_generator.generate_random_vector(4*32);
  auto _b_hh2_vec = random_generator.generate_random_vector(4*32);
  auto _w_ih2 = hypertea::TensorGPU<float>(_w_ih2_vec); 
  auto _w_hh2 = hypertea::TensorGPU<float>(_w_hh2_vec); 
  auto _b_ih2 = hypertea::TensorGPU<float>(_b_ih2_vec); 
  auto _b_hh2 = hypertea::TensorGPU<float>(_b_hh2_vec); 

  auto input_tensor = hypertea::TensorGPU<float>(random_generator.generate_random_vector(5*64));
  auto hidden_tensor0 = hypertea::TensorGPU<float>(random_generator.generate_random_vector(2*32));
  auto hidden_tensor1 = hypertea::TensorGPU<float>(random_generator.generate_random_vector(2*32));
  auto hidden_tensor2 = hypertea::TensorGPU<float>(random_generator.generate_random_vector(2*32));
  auto output_tensor = hypertea::TensorGPU<float>(32);

  hypertea::StackedRNN_GPU<float> lstm( std::vector<hypertea::RNNOp_GPU<float>* >{
      new hypertea::UnidirectionalRNN_GPU<float> (
        64, 32, 
        _w_ih0.mutable_data(), _w_hh0.mutable_data(), 
        _b_ih0.mutable_data(), _b_hh0.mutable_data(), 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      ),

      new hypertea::UnidirectionalRNN_GPU<float> (
        32, 32, 
        _w_ih1.mutable_data(), _w_hh1.mutable_data(), 
        _b_ih1.mutable_data(), _b_hh1.mutable_data(), 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      ),
      new hypertea::UnidirectionalRNN_GPU<float> (
        32, 32, 
        _w_ih2.mutable_data(), _w_hh2.mutable_data(), 
        _b_ih2.mutable_data(), _b_hh2.mutable_data(), 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      )
    }
  );


  output_tensor = lstm.Forward(
    input_tensor,
    std::vector<hypertea::TensorGPU<float> > {hidden_tensor0, hidden_tensor1, hidden_tensor2}
  );

  const float* output_data = output_tensor.debug_cpu_data();
  for (int i = 0; i < test_result::uni_multi3_lstm_result.size(); ++i) {
    EXPECT_NEAR(output_data[i], test_result::uni_multi3_lstm_result[i], 1e-3);
  }

}




TYPED_TEST(RNNTestGPU, test_bi_multi3_lstm_GPU) {
    
  typedef typename TypeParam::Dtype Dtype;

  fake_random_number random_generator;

  auto _w_ih0_vec = random_generator.generate_random_vector(4*32*64);
  auto _w_hh0_vec = random_generator.generate_random_vector(4*32*32);
  auto _b_ih0_vec = random_generator.generate_random_vector(4*32);
  auto _b_hh0_vec = random_generator.generate_random_vector(4*32);
  auto r_w_ih0_vec = random_generator.generate_random_vector(4*32*64);
  auto r_w_hh0_vec = random_generator.generate_random_vector(4*32*32);
  auto r_b_ih0_vec = random_generator.generate_random_vector(4*32);
  auto r_b_hh0_vec = random_generator.generate_random_vector(4*32);
  auto _w_ih0 = hypertea::TensorGPU<float>(_w_ih0_vec); 
  auto _w_hh0 = hypertea::TensorGPU<float>(_w_hh0_vec); 
  auto _b_ih0 = hypertea::TensorGPU<float>(_b_ih0_vec); 
  auto _b_hh0 = hypertea::TensorGPU<float>(_b_hh0_vec);
  auto r_w_ih0 = hypertea::TensorGPU<float>(r_w_ih0_vec); 
  auto r_w_hh0 = hypertea::TensorGPU<float>(r_w_hh0_vec); 
  auto r_b_ih0 = hypertea::TensorGPU<float>(r_b_ih0_vec); 
  auto r_b_hh0 = hypertea::TensorGPU<float>(r_b_hh0_vec);


  auto _w_ih1_vec = random_generator.generate_random_vector(4*32*64);
  auto _w_hh1_vec = random_generator.generate_random_vector(4*32*32);
  auto _b_ih1_vec = random_generator.generate_random_vector(4*32);
  auto _b_hh1_vec = random_generator.generate_random_vector(4*32);
  auto r_w_ih1_vec = random_generator.generate_random_vector(4*32*64);
  auto r_w_hh1_vec = random_generator.generate_random_vector(4*32*32);
  auto r_b_ih1_vec = random_generator.generate_random_vector(4*32);
  auto r_b_hh1_vec = random_generator.generate_random_vector(4*32);
  auto _w_ih1 = hypertea::TensorGPU<float>(_w_ih1_vec); 
  auto _w_hh1 = hypertea::TensorGPU<float>(_w_hh1_vec); 
  auto _b_ih1 = hypertea::TensorGPU<float>(_b_ih1_vec); 
  auto _b_hh1 = hypertea::TensorGPU<float>(_b_hh1_vec);
  auto r_w_ih1 = hypertea::TensorGPU<float>(r_w_ih1_vec); 
  auto r_w_hh1 = hypertea::TensorGPU<float>(r_w_hh1_vec); 
  auto r_b_ih1 = hypertea::TensorGPU<float>(r_b_ih1_vec); 
  auto r_b_hh1 = hypertea::TensorGPU<float>(r_b_hh1_vec);

  auto _w_ih2_vec = random_generator.generate_random_vector(4*32*64);
  auto _w_hh2_vec = random_generator.generate_random_vector(4*32*32);
  auto _b_ih2_vec = random_generator.generate_random_vector(4*32);
  auto _b_hh2_vec = random_generator.generate_random_vector(4*32);
  auto r_w_ih2_vec = random_generator.generate_random_vector(4*32*64);
  auto r_w_hh2_vec = random_generator.generate_random_vector(4*32*32);
  auto r_b_ih2_vec = random_generator.generate_random_vector(4*32);
  auto r_b_hh2_vec = random_generator.generate_random_vector(4*32);
  auto _w_ih2 = hypertea::TensorGPU<float>(_w_ih2_vec); 
  auto _w_hh2 = hypertea::TensorGPU<float>(_w_hh2_vec); 
  auto _b_ih2 = hypertea::TensorGPU<float>(_b_ih2_vec); 
  auto _b_hh2 = hypertea::TensorGPU<float>(_b_hh2_vec);
  auto r_w_ih2 = hypertea::TensorGPU<float>(r_w_ih2_vec); 
  auto r_w_hh2 = hypertea::TensorGPU<float>(r_w_hh2_vec); 
  auto r_b_ih2 = hypertea::TensorGPU<float>(r_b_ih2_vec); 
  auto r_b_hh2 = hypertea::TensorGPU<float>(r_b_hh2_vec);

  auto input_tensor = hypertea::TensorGPU<float>(random_generator.generate_random_vector(5*64));
  auto hidden_tensor0 = hypertea::TensorGPU<float>(random_generator.generate_random_vector(4*32));
  auto hidden_tensor1 = hypertea::TensorGPU<float>(random_generator.generate_random_vector(4*32));
  auto hidden_tensor2 = hypertea::TensorGPU<float>(random_generator.generate_random_vector(4*32));
  auto output_tensor = hypertea::TensorGPU<float>(32);

  hypertea::StackedRNN_GPU<float> lstm( std::vector<hypertea::RNNOp_GPU<float>* >{
      new hypertea::BidirectionalRNN_GPU<float> (
        64, 32, 
        _w_ih0.mutable_data(), r_w_ih0.mutable_data(), 
        _w_hh0.mutable_data(), r_w_hh0.mutable_data(), 
        _b_ih0.mutable_data(), r_b_ih0.mutable_data(), 
        _b_hh0.mutable_data(), r_b_hh0.mutable_data(), 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      ),
      new hypertea::BidirectionalRNN_GPU<float> (
        64, 32, 
        _w_ih1.mutable_data(), r_w_ih1.mutable_data(), 
        _w_hh1.mutable_data(), r_w_hh1.mutable_data(), 
        _b_ih1.mutable_data(), r_b_ih1.mutable_data(), 
        _b_hh1.mutable_data(), r_b_hh1.mutable_data(), 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      ),      
      new hypertea::BidirectionalRNN_GPU<float> (
        64, 32, 
        _w_ih2.mutable_data(), r_w_ih2.mutable_data(), 
        _w_hh2.mutable_data(), r_w_hh2.mutable_data(), 
        _b_ih2.mutable_data(), r_b_ih2.mutable_data(), 
        _b_hh2.mutable_data(), r_b_hh2.mutable_data(), 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      )
    }
  );


  output_tensor = lstm.Forward(
    input_tensor,
    std::vector<hypertea::TensorGPU<float> > {hidden_tensor0, hidden_tensor1, hidden_tensor2}
  );



  const float* output_data = output_tensor.debug_cpu_data();

  for (int i = 0; i < test_result::bi_multi3_lstm_result.size(); ++i) {
    EXPECT_NEAR(output_data[i], test_result::bi_multi3_lstm_result[i], 1e-3);
  }

}



}  // namespace caffe
