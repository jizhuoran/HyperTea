#include <algorithm>
#include <vector>

#include "gtest/gtest.h"


#include "hypertea/common.hpp"

#include "test_hypertea_util.hpp"
#include "hypertea/operators/rnn_op.hpp"

#include "test_result/rnn_result.hpp"

namespace hypertea {


template <typename TypeParam>
class RNN_Test : public ::testing::Test {
 public:
  // using DeviceTensor = TypeParam;
 protected:
  RNN_Test() {
#ifdef USE_OPENCL    
    hypertea::OpenCLHandler::Get().build_opencl_math_code(false);
#endif
  }
  virtual ~RNN_Test() {}
};



TYPED_TEST_CASE(RNN_Test, TestDtypes);


 
TYPED_TEST(RNN_Test, test_uni_single_gru) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  auto _w_ih0 = DeviceTensor(random_generator.generate_random_vector(3*32*64));
  auto _w_hh0 = DeviceTensor(random_generator.generate_random_vector(3*32*32));
  auto _b_ih0 = DeviceTensor(random_generator.generate_random_vector(3*32));
  auto _b_hh0 = DeviceTensor(random_generator.generate_random_vector(3*32));


  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(5*64));
  auto hidden_tensor = DeviceTensor(random_generator.generate_random_vector(1*32));
  auto output_tensor = DeviceTensor(32);

  hypertea::StackedRNN<DeviceTensor> gru( std::vector<hypertea::RNNOp<DeviceTensor>* >{
      new hypertea::UnidirectionalRNN<DeviceTensor> (
        64, 32, 
        _w_ih0, _w_hh0, 
        _b_ih0, _b_hh0, 
        hypertea::RNN_CELL_TYPE::GRU_CELL
      )
    }
  );

  output_tensor = gru.Forward(
    input_tensor,
    std::vector<DeviceTensor > {hidden_tensor}
  );

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::uni_single_gru_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::uni_single_gru_result[i], 1e-3);
  }
}


TYPED_TEST(RNN_Test, test_bi_single_gru) {
  
  using DeviceTensor = TypeParam;
    
  fake_random_number random_generator;

  auto _w_ih0 = DeviceTensor(random_generator.generate_random_vector(3*32*64));
  auto _w_hh0 = DeviceTensor(random_generator.generate_random_vector(3*32*32));
  auto _b_ih0 = DeviceTensor(random_generator.generate_random_vector(3*32));
  auto _b_hh0 = DeviceTensor(random_generator.generate_random_vector(3*32));

  auto r_w_ih0 = DeviceTensor(random_generator.generate_random_vector(3*32*64));
  auto r_w_hh0 = DeviceTensor(random_generator.generate_random_vector(3*32*32));
  auto r_b_ih0 = DeviceTensor(random_generator.generate_random_vector(3*32));
  auto r_b_hh0 = DeviceTensor(random_generator.generate_random_vector(3*32));

  auto input_tensor = DeviceTensor(DeviceTensor(random_generator.generate_random_vector(5*64)));
  auto hidden_tensor = DeviceTensor(DeviceTensor(random_generator.generate_random_vector(2*32)));
  auto output_tensor = DeviceTensor(32);

  hypertea::StackedRNN<DeviceTensor> gru( std::vector<hypertea::RNNOp<DeviceTensor>* >{
      new hypertea::BidirectionalRNN<DeviceTensor> (
        64, 32, 
        _w_ih0, r_w_ih0, 
        _w_hh0, r_w_hh0, 
        _b_ih0, r_b_ih0, 
        _b_hh0, r_b_hh0, 
        hypertea::RNN_CELL_TYPE::GRU_CELL
      )
    }
  );


  output_tensor = gru.Forward(
    input_tensor,
    std::vector<DeviceTensor > {hidden_tensor}
  );


  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bi_single_gru_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bi_single_gru_result[i], 1e-3);
  }
}


TYPED_TEST(RNN_Test, test_uni_multi3_gru) {
  
  using DeviceTensor = TypeParam;
    
  fake_random_number random_generator;

  auto _w_ih0 = DeviceTensor(random_generator.generate_random_vector(3*32*64));
  auto _w_hh0 = DeviceTensor(random_generator.generate_random_vector(3*32*32));
  auto _b_ih0 = DeviceTensor(random_generator.generate_random_vector(3*32));
  auto _b_hh0 = DeviceTensor(random_generator.generate_random_vector(3*32));

  auto _w_ih1 = DeviceTensor(random_generator.generate_random_vector(3*32*32));
  auto _w_hh1 = DeviceTensor(random_generator.generate_random_vector(3*32*32));
  auto _b_ih1 = DeviceTensor(random_generator.generate_random_vector(3*32));
  auto _b_hh1 = DeviceTensor(random_generator.generate_random_vector(3*32));


  auto _w_ih2 = DeviceTensor(random_generator.generate_random_vector(3*32*32));
  auto _w_hh2 = DeviceTensor(random_generator.generate_random_vector(3*32*32));
  auto _b_ih2 = DeviceTensor(random_generator.generate_random_vector(3*32));
  auto _b_hh2 = DeviceTensor(random_generator.generate_random_vector(3*32));


  auto input_tensor = DeviceTensor(DeviceTensor(random_generator.generate_random_vector(5*64)));
  auto hidden_tensor0 = DeviceTensor(DeviceTensor(random_generator.generate_random_vector(1*32)));
  auto hidden_tensor1 = DeviceTensor(DeviceTensor(random_generator.generate_random_vector(1*32)));
  auto hidden_tensor2 = DeviceTensor(DeviceTensor(random_generator.generate_random_vector(1*32)));
  auto output_tensor = DeviceTensor(32);

  hypertea::StackedRNN<DeviceTensor> gru( std::vector<hypertea::RNNOp<DeviceTensor>* >{
    new hypertea::UnidirectionalRNN<DeviceTensor> (
      64, 32, 
      _w_ih0, _w_hh0, 
      _b_ih0, _b_hh0, 
      hypertea::RNN_CELL_TYPE::GRU_CELL
    ),

    new hypertea::UnidirectionalRNN<DeviceTensor> (
      32, 32, 
      _w_ih1, _w_hh1, 
      _b_ih1, _b_hh1, 
      hypertea::RNN_CELL_TYPE::GRU_CELL
    ),
    new hypertea::UnidirectionalRNN<DeviceTensor> (
      32, 32, 
      _w_ih2, _w_hh2, 
      _b_ih2, _b_hh2, 
      hypertea::RNN_CELL_TYPE::GRU_CELL
    )
    }
  );


  output_tensor = gru.Forward(
    input_tensor,
    std::vector<DeviceTensor > {hidden_tensor0, hidden_tensor1, hidden_tensor2}
  );


  auto output_data = output_tensor.debug_gtest_cpu_data();
  for (int i = 0; i < test_result::uni_multi3_gru_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::uni_multi3_gru_result[i], 1e-3);
  }

}


TYPED_TEST(RNN_Test, test_bi_multi3_gru) {

  using DeviceTensor = TypeParam;

  fake_random_number random_generator;

  auto _w_ih0 = DeviceTensor(random_generator.generate_random_vector(3*32*64));
  auto _w_hh0 = DeviceTensor(random_generator.generate_random_vector(3*32*32));
  auto _b_ih0 = DeviceTensor(random_generator.generate_random_vector(3*32));
  auto _b_hh0 = DeviceTensor(random_generator.generate_random_vector(3*32));
  auto r_w_ih0 = DeviceTensor(random_generator.generate_random_vector(3*32*64));
  auto r_w_hh0 = DeviceTensor(random_generator.generate_random_vector(3*32*32));
  auto r_b_ih0 = DeviceTensor(random_generator.generate_random_vector(3*32));
  auto r_b_hh0 = DeviceTensor(random_generator.generate_random_vector(3*32));



  auto _w_ih1 = DeviceTensor(random_generator.generate_random_vector(3*32*64));
  auto _w_hh1 = DeviceTensor(random_generator.generate_random_vector(3*32*32));
  auto _b_ih1 = DeviceTensor(random_generator.generate_random_vector(3*32));
  auto _b_hh1 = DeviceTensor(random_generator.generate_random_vector(3*32));
  auto r_w_ih1 = DeviceTensor(random_generator.generate_random_vector(3*32*64));
  auto r_w_hh1 = DeviceTensor(random_generator.generate_random_vector(3*32*32));
  auto r_b_ih1 = DeviceTensor(random_generator.generate_random_vector(3*32));
  auto r_b_hh1 = DeviceTensor(random_generator.generate_random_vector(3*32));


  auto _w_ih2 = DeviceTensor(random_generator.generate_random_vector(3*32*64));
  auto _w_hh2 = DeviceTensor(random_generator.generate_random_vector(3*32*32));
  auto _b_ih2 = DeviceTensor(random_generator.generate_random_vector(3*32));
  auto _b_hh2 = DeviceTensor(random_generator.generate_random_vector(3*32));
  auto r_w_ih2 = DeviceTensor(random_generator.generate_random_vector(3*32*64));
  auto r_w_hh2 = DeviceTensor(random_generator.generate_random_vector(3*32*32));
  auto r_b_ih2 = DeviceTensor(random_generator.generate_random_vector(3*32));
  auto r_b_hh2 = DeviceTensor(random_generator.generate_random_vector(3*32));


  auto input_tensor = DeviceTensor(DeviceTensor(random_generator.generate_random_vector(5*64)));
  auto hidden_tensor0 = DeviceTensor(DeviceTensor(random_generator.generate_random_vector(2*32)));
  auto hidden_tensor1 = DeviceTensor(DeviceTensor(random_generator.generate_random_vector(2*32)));
  auto hidden_tensor2 = DeviceTensor(DeviceTensor(random_generator.generate_random_vector(2*32)));
  auto output_tensor = DeviceTensor(32);

  hypertea::StackedRNN<DeviceTensor> gru( std::vector<hypertea::RNNOp<DeviceTensor>* >{

      new hypertea::BidirectionalRNN<DeviceTensor> (
        64, 32, 
        _w_ih0, r_w_ih0, 
        _w_hh0, r_w_hh0, 
        _b_ih0, r_b_ih0, 
        _b_hh0, r_b_hh0, 
        hypertea::RNN_CELL_TYPE::GRU_CELL
      ),
      new hypertea::BidirectionalRNN<DeviceTensor> (
        64, 32, 
        _w_ih1, r_w_ih1, 
        _w_hh1, r_w_hh1, 
        _b_ih1, r_b_ih1, 
        _b_hh1, r_b_hh1, 
        hypertea::RNN_CELL_TYPE::GRU_CELL
      ),      
      new hypertea::BidirectionalRNN<DeviceTensor> (
        64, 32, 
        _w_ih2, r_w_ih2, 
        _w_hh2, r_w_hh2, 
        _b_ih2, r_b_ih2, 
        _b_hh2, r_b_hh2, 
        hypertea::RNN_CELL_TYPE::GRU_CELL
      )
    }
  );


  output_tensor = gru.Forward(
    input_tensor,
    std::vector<DeviceTensor > {hidden_tensor0, hidden_tensor1, hidden_tensor2}
  );

  auto output_data = output_tensor.debug_gtest_cpu_data();
  for (int i = 0; i < test_result::bi_multi3_gru_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bi_multi3_gru_result[i], 1e-3);
  }

}





TYPED_TEST(RNN_Test, test_uni_single_lstm) {

  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  auto _w_ih0 = DeviceTensor(random_generator.generate_random_vector(4*32*64));
  auto _w_hh0 = DeviceTensor(random_generator.generate_random_vector(4*32*32));
  auto _b_ih0 = DeviceTensor(random_generator.generate_random_vector(4*32));
  auto _b_hh0 = DeviceTensor(random_generator.generate_random_vector(4*32));


  auto input_tensor = DeviceTensor(DeviceTensor(random_generator.generate_random_vector(5*64)));
  auto hidden_tensor = DeviceTensor(DeviceTensor(random_generator.generate_random_vector(2*32)));
  auto output_tensor = DeviceTensor(32);

  hypertea::StackedRNN<DeviceTensor> lstm( std::vector<hypertea::RNNOp<DeviceTensor>* >{
      new hypertea::UnidirectionalRNN<DeviceTensor> (
        64, 32, 
        _w_ih0, _w_hh0, 
        _b_ih0, _b_hh0, 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      )
    }
  );


  output_tensor = lstm.Forward(
    input_tensor,
    std::vector<DeviceTensor > {hidden_tensor}
  );

  auto output_data = output_tensor.debug_gtest_cpu_data();
  for (int i = 0; i < test_result::uni_single_lstm_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::uni_single_lstm_result[i], 1e-3);
  }

}


TYPED_TEST(RNN_Test, test_bi_single_lstm) {

  using DeviceTensor = TypeParam;

  fake_random_number random_generator;

  auto _w_ih0 = DeviceTensor(random_generator.generate_random_vector(4*32*64));
  auto _w_hh0 = DeviceTensor(random_generator.generate_random_vector(4*32*32));
  auto _b_ih0 = DeviceTensor(random_generator.generate_random_vector(4*32));
  auto _b_hh0 = DeviceTensor(random_generator.generate_random_vector(4*32));

  auto r_w_ih0 = DeviceTensor(random_generator.generate_random_vector(4*32*64));
  auto r_w_hh0 = DeviceTensor(random_generator.generate_random_vector(4*32*32));
  auto r_b_ih0 = DeviceTensor(random_generator.generate_random_vector(4*32));
  auto r_b_hh0 = DeviceTensor(random_generator.generate_random_vector(4*32));

  auto input_tensor = DeviceTensor(DeviceTensor(random_generator.generate_random_vector(5*64)));
  auto hidden_tensor = DeviceTensor(DeviceTensor(random_generator.generate_random_vector(4*32)));
  auto output_tensor = DeviceTensor(32);

  hypertea::StackedRNN<DeviceTensor> lstm( std::vector<hypertea::RNNOp<DeviceTensor>* >{
      new hypertea::BidirectionalRNN<DeviceTensor> (
        64, 32, 
        _w_ih0, r_w_ih0, 
        _w_hh0, r_w_hh0, 
        _b_ih0, r_b_ih0, 
        _b_hh0, r_b_hh0, 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      )
    }
  );


  output_tensor = lstm.Forward(
    input_tensor,
    std::vector<DeviceTensor > {hidden_tensor}
  );


  auto output_data = output_tensor.debug_gtest_cpu_data();
  for (int i = 0; i < test_result::bi_single_lstm_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bi_single_lstm_result[i], 1e-3);
  }

}



TYPED_TEST(RNN_Test, test_uni_multi3_lstm) {

  using DeviceTensor = TypeParam;

  fake_random_number random_generator;

  auto _w_ih0 = DeviceTensor(random_generator.generate_random_vector(4*32*64));
  auto _w_hh0 = DeviceTensor(random_generator.generate_random_vector(4*32*32));
  auto _b_ih0 = DeviceTensor(random_generator.generate_random_vector(4*32));
  auto _b_hh0 = DeviceTensor(random_generator.generate_random_vector(4*32));

  auto _w_ih1 = DeviceTensor(random_generator.generate_random_vector(4*32*32));
  auto _w_hh1 = DeviceTensor(random_generator.generate_random_vector(4*32*32));
  auto _b_ih1 = DeviceTensor(random_generator.generate_random_vector(4*32));
  auto _b_hh1 = DeviceTensor(random_generator.generate_random_vector(4*32));


  auto _w_ih2 = DeviceTensor(random_generator.generate_random_vector(4*32*32));
  auto _w_hh2 = DeviceTensor(random_generator.generate_random_vector(4*32*32));
  auto _b_ih2 = DeviceTensor(random_generator.generate_random_vector(4*32));
  auto _b_hh2 = DeviceTensor(random_generator.generate_random_vector(4*32));


  auto input_tensor = DeviceTensor(DeviceTensor(random_generator.generate_random_vector(5*64)));
  auto hidden_tensor0 = DeviceTensor(DeviceTensor(random_generator.generate_random_vector(2*32)));
  auto hidden_tensor1 = DeviceTensor(DeviceTensor(random_generator.generate_random_vector(2*32)));
  auto hidden_tensor2 = DeviceTensor(DeviceTensor(random_generator.generate_random_vector(2*32)));
  auto output_tensor = DeviceTensor(32);

  hypertea::StackedRNN<DeviceTensor> lstm( std::vector<hypertea::RNNOp<DeviceTensor>* >{
      new hypertea::UnidirectionalRNN<DeviceTensor> (
        64, 32, 
        _w_ih0, _w_hh0, 
        _b_ih0, _b_hh0, 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      ),

      new hypertea::UnidirectionalRNN<DeviceTensor> (
        32, 32, 
        _w_ih1, _w_hh1, 
        _b_ih1, _b_hh1, 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      ),
      new hypertea::UnidirectionalRNN<DeviceTensor> (
        32, 32, 
        _w_ih2, _w_hh2, 
        _b_ih2, _b_hh2, 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      )
    }
  );


  output_tensor = lstm.Forward(
    input_tensor,
    std::vector<DeviceTensor > {hidden_tensor0, hidden_tensor1, hidden_tensor2}
  );

  auto output_data = output_tensor.debug_gtest_cpu_data();
  for (int i = 0; i < test_result::uni_multi3_lstm_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::uni_multi3_lstm_result[i], 1e-3);
  }

}




TYPED_TEST(RNN_Test, test_bi_multi3_lstm) {

  using DeviceTensor = TypeParam;

  fake_random_number random_generator;

  auto _w_ih0 = DeviceTensor(random_generator.generate_random_vector(4*32*64));
  auto _w_hh0 = DeviceTensor(random_generator.generate_random_vector(4*32*32));
  auto _b_ih0 = DeviceTensor(random_generator.generate_random_vector(4*32));
  auto _b_hh0 = DeviceTensor(random_generator.generate_random_vector(4*32));
  auto r_w_ih0 = DeviceTensor(random_generator.generate_random_vector(4*32*64));
  auto r_w_hh0 = DeviceTensor(random_generator.generate_random_vector(4*32*32));
  auto r_b_ih0 = DeviceTensor(random_generator.generate_random_vector(4*32));
  auto r_b_hh0 = DeviceTensor(random_generator.generate_random_vector(4*32));



  auto _w_ih1 = DeviceTensor(random_generator.generate_random_vector(4*32*64));
  auto _w_hh1 = DeviceTensor(random_generator.generate_random_vector(4*32*32));
  auto _b_ih1 = DeviceTensor(random_generator.generate_random_vector(4*32));
  auto _b_hh1 = DeviceTensor(random_generator.generate_random_vector(4*32));
  auto r_w_ih1 = DeviceTensor(random_generator.generate_random_vector(4*32*64));
  auto r_w_hh1 = DeviceTensor(random_generator.generate_random_vector(4*32*32));
  auto r_b_ih1 = DeviceTensor(random_generator.generate_random_vector(4*32));
  auto r_b_hh1 = DeviceTensor(random_generator.generate_random_vector(4*32));


  auto _w_ih2 = DeviceTensor(random_generator.generate_random_vector(4*32*64));
  auto _w_hh2 = DeviceTensor(random_generator.generate_random_vector(4*32*32));
  auto _b_ih2 = DeviceTensor(random_generator.generate_random_vector(4*32));
  auto _b_hh2 = DeviceTensor(random_generator.generate_random_vector(4*32));
  auto r_w_ih2 = DeviceTensor(random_generator.generate_random_vector(4*32*64));
  auto r_w_hh2 = DeviceTensor(random_generator.generate_random_vector(4*32*32));
  auto r_b_ih2 = DeviceTensor(random_generator.generate_random_vector(4*32));
  auto r_b_hh2 = DeviceTensor(random_generator.generate_random_vector(4*32));


  auto input_tensor = DeviceTensor(DeviceTensor(random_generator.generate_random_vector(5*64)));
  auto hidden_tensor0 = DeviceTensor(DeviceTensor(random_generator.generate_random_vector(4*32)));
  auto hidden_tensor1 = DeviceTensor(DeviceTensor(random_generator.generate_random_vector(4*32)));
  auto hidden_tensor2 = DeviceTensor(DeviceTensor(random_generator.generate_random_vector(4*32)));
  auto output_tensor = DeviceTensor(32);

  hypertea::StackedRNN<DeviceTensor> lstm( std::vector<hypertea::RNNOp<DeviceTensor>* >{
      new hypertea::BidirectionalRNN<DeviceTensor> (
        64, 32, 
        _w_ih0, r_w_ih0, 
        _w_hh0, r_w_hh0, 
        _b_ih0, r_b_ih0, 
        _b_hh0, r_b_hh0, 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      ),
      new hypertea::BidirectionalRNN<DeviceTensor> (
        64, 32, 
        _w_ih1, r_w_ih1, 
        _w_hh1, r_w_hh1, 
        _b_ih1, r_b_ih1, 
        _b_hh1, r_b_hh1, 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      ),      
      new hypertea::BidirectionalRNN<DeviceTensor> (
        64, 32, 
        _w_ih2, r_w_ih2, 
        _w_hh2, r_w_hh2, 
        _b_ih2, r_b_ih2, 
        _b_hh2, r_b_hh2, 
        hypertea::RNN_CELL_TYPE::LSTM_CELL
      )
    }
  );


  output_tensor = lstm.Forward(
    input_tensor,
    std::vector<DeviceTensor > {hidden_tensor0, hidden_tensor1, hidden_tensor2}
  );

  auto output_data = output_tensor.debug_gtest_cpu_data();
  for (int i = 0; i < test_result::bi_multi3_lstm_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bi_multi3_lstm_result[i], 1e-3);
  }

}






}  // namespace caffe
