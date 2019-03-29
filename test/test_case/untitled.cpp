TYPED_TEST(RNNTestGPU, test_uni_single_gru_GPU() {
    
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


TYPED_TEST(RNNTestGPU, test_bi_single_gru_GPU() {
    
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



TYPED_TEST(RNNTestGPU, test_uni_multi3_gru_GPU() {
    
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




TYPED_TEST(RNNTestGPU, test_bi_multi3_gru_GPU() {
    
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

TYPED_TEST(RNNTestGPU, test_gru_GPU() {
  
  typedef typename TypeParam::Dtype Dtype;

  hypertea::OpenCLHandler::Get().build_opencl_math_code(false);

  test_uni_single_gru_GPU();
  test_bi_single_gru_GPU();
  test_uni_multi3_gru_GPU();
  test_bi_multi3_gru_GPU();
}



TYPED_TEST(RNNTestGPU, test_uni_single_lstm_GPU() {
    
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


TYPED_TEST(RNNTestGPU, test_bi_single_lstm_GPU() {
    
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



TYPED_TEST(RNNTestGPU, test_uni_multi3_lstm_GPU() {
    
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




TYPED_TEST(RNNTestGPU, test_bi_multi3_lstm_GPU() {
    
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