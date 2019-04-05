#include <algorithm>
#include <vector>

#include "gtest/gtest.h"


#include "hypertea/common.hpp"

#include "test_hypertea_util.hpp"
#include "hypertea/operators/conv_op.hpp"
#include "hypertea/operators/deconv_op.hpp"

#include "test_result/conv_result.hpp"
#include "test_result/deconv_result.hpp"

namespace hypertea { 

 
template <typename TypeParam>
class CONVTestCPU : public ::testing::Test {
 public:
  typedef typename TypeParam::Dtype Dtype;
 protected:
  CONVTestCPU() {}
  virtual ~CONVTestCPU() {}
};

template <typename TypeParam>
class CONVTestGPU : public ::testing::Test {
 public:
  typedef typename TypeParam::Dtype Dtype;
 protected:
  CONVTestGPU() {
    hypertea::OpenCLHandler::Get().build_opencl_math_code(false);
  }
  virtual ~CONVTestGPU() {}
};


TYPED_TEST_CASE(CONVTestCPU, TestDtypesCPU);
TYPED_TEST_CASE(CONVTestGPU, TestDtypesGPU);


TYPED_TEST(CONVTestCPU, test_conv_2_3_1_1_0_1_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(6));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(256));
  
  hypertea::ConvolutionOp_CPU<float> convolutional = ConvolutionOp_CPU<float>(weight.mutable_data(), bias.mutable_data(), 1, true, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {0,0}, std::vector<int> {1,1}, std::vector<int> {2,2,8,8}, std::vector<int> {2,3,8,8}, false);


  auto output_tensor = convolutional.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::conv_2_3_1_1_0_1_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::conv_2_3_1_1_0_1_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONVTestCPU, test_conv_2_3_1_2_2_3_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(6));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(256));
  
  hypertea::ConvolutionOp_CPU<float> convolutional = ConvolutionOp_CPU<float>(weight.mutable_data(), bias.mutable_data(), 1, false, std::vector<int> {1,1}, std::vector<int> {2,2}, std::vector<int> {2,2}, std::vector<int> {3,3}, std::vector<int> {2,2,8,8}, std::vector<int> {2,3,6,6}, false);


  auto output_tensor = convolutional.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::conv_2_3_1_2_2_3_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::conv_2_3_1_2_2_3_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONVTestCPU, test_conv_2_3_3_1_0_1_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(54));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(256));
  
  hypertea::ConvolutionOp_CPU<float> convolutional = ConvolutionOp_CPU<float>(weight.mutable_data(), bias.mutable_data(), 1, false, std::vector<int> {3,3}, std::vector<int> {1,1}, std::vector<int> {0,0}, std::vector<int> {1,1}, std::vector<int> {2,2,8,8}, std::vector<int> {2,3,6,6}, false);


  auto output_tensor = convolutional.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::conv_2_3_3_1_0_1_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::conv_2_3_3_1_0_1_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONVTestCPU, test_conv_2_3_3_2_2_3_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(54));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(256));
  
  hypertea::ConvolutionOp_CPU<float> convolutional = ConvolutionOp_CPU<float>(weight.mutable_data(), bias.mutable_data(), 1, false, std::vector<int> {3,3}, std::vector<int> {2,2}, std::vector<int> {2,2}, std::vector<int> {3,3}, std::vector<int> {2,2,8,8}, std::vector<int> {2,3,3,3}, false);


  auto output_tensor = convolutional.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::conv_2_3_3_2_2_3_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::conv_2_3_3_2_2_3_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONVTestCPU, test_conv_4_3_1_1_0_1_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(12));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(512));
  
  hypertea::ConvolutionOp_CPU<float> convolutional = ConvolutionOp_CPU<float>(weight.mutable_data(), bias.mutable_data(), 1, true, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {0,0}, std::vector<int> {1,1}, std::vector<int> {2,4,8,8}, std::vector<int> {2,3,8,8}, false);


  auto output_tensor = convolutional.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::conv_4_3_1_1_0_1_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::conv_4_3_1_1_0_1_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONVTestCPU, test_conv_4_3_1_2_2_3_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(12));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(512));
  
  hypertea::ConvolutionOp_CPU<float> convolutional = ConvolutionOp_CPU<float>(weight.mutable_data(), bias.mutable_data(), 1, false, std::vector<int> {1,1}, std::vector<int> {2,2}, std::vector<int> {2,2}, std::vector<int> {3,3}, std::vector<int> {2,4,8,8}, std::vector<int> {2,3,6,6}, false);


  auto output_tensor = convolutional.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::conv_4_3_1_2_2_3_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::conv_4_3_1_2_2_3_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONVTestCPU, test_conv_4_3_3_1_0_1_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(108));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(512));
  
  hypertea::ConvolutionOp_CPU<float> convolutional = ConvolutionOp_CPU<float>(weight.mutable_data(), bias.mutable_data(), 1, false, std::vector<int> {3,3}, std::vector<int> {1,1}, std::vector<int> {0,0}, std::vector<int> {1,1}, std::vector<int> {2,4,8,8}, std::vector<int> {2,3,6,6}, false);


  auto output_tensor = convolutional.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::conv_4_3_3_1_0_1_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::conv_4_3_3_1_0_1_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONVTestCPU, test_conv_4_3_3_2_2_3_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(108));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(512));
  
  hypertea::ConvolutionOp_CPU<float> convolutional = ConvolutionOp_CPU<float>(weight.mutable_data(), bias.mutable_data(), 1, false, std::vector<int> {3,3}, std::vector<int> {2,2}, std::vector<int> {2,2}, std::vector<int> {3,3}, std::vector<int> {2,4,8,8}, std::vector<int> {2,3,3,3}, false);


  auto output_tensor = convolutional.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::conv_4_3_3_2_2_3_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::conv_4_3_3_2_2_3_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONVTestCPU, test_deconv_2_2_1_1_0_1_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(4));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(2));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(256));
  
  hypertea::DeconvolutionOp_CPU<float> convolutional = DeconvolutionOp_CPU<float>(weight.mutable_data(), bias.mutable_data(), 1, true, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {0,0}, std::vector<int> {1,1}, std::vector<int> {2,2,8,8}, std::vector<int> {2,2,8,8}, false);


  auto output_tensor = convolutional.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_2_2_1_1_0_1_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_2_2_1_1_0_1_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONVTestCPU, test_deconv_2_2_1_2_2_3_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(4));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(2));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(256));
  
  hypertea::DeconvolutionOp_CPU<float> convolutional = DeconvolutionOp_CPU<float>(weight.mutable_data(), bias.mutable_data(), 1, false, std::vector<int> {1,1}, std::vector<int> {2,2}, std::vector<int> {2,2}, std::vector<int> {3,3}, std::vector<int> {2,2,8,8}, std::vector<int> {2,2,11,11}, false);


  auto output_tensor = convolutional.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_2_2_1_2_2_3_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_2_2_1_2_2_3_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONVTestCPU, test_deconv_2_2_3_1_0_1_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(36));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(2));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(256));
  
  hypertea::DeconvolutionOp_CPU<float> convolutional = DeconvolutionOp_CPU<float>(weight.mutable_data(), bias.mutable_data(), 1, false, std::vector<int> {3,3}, std::vector<int> {1,1}, std::vector<int> {0,0}, std::vector<int> {1,1}, std::vector<int> {2,2,8,8}, std::vector<int> {2,2,10,10}, false);


  auto output_tensor = convolutional.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_2_2_3_1_0_1_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_2_2_3_1_0_1_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONVTestCPU, test_deconv_2_2_3_2_2_3_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(36));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(2));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(256));
  
  hypertea::DeconvolutionOp_CPU<float> convolutional = DeconvolutionOp_CPU<float>(weight.mutable_data(), bias.mutable_data(), 1, false, std::vector<int> {3,3}, std::vector<int> {2,2}, std::vector<int> {2,2}, std::vector<int> {3,3}, std::vector<int> {2,2,8,8}, std::vector<int> {2,2,17,17}, false);


  auto output_tensor = convolutional.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_2_2_3_2_2_3_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_2_2_3_2_2_3_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONVTestCPU, test_deconv_3_4_1_1_0_1_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(12));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(4));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(384));
  
  hypertea::DeconvolutionOp_CPU<float> convolutional = DeconvolutionOp_CPU<float>(weight.mutable_data(), bias.mutable_data(), 1, true, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {0,0}, std::vector<int> {1,1}, std::vector<int> {2,3,8,8}, std::vector<int> {2,4,8,8}, false);


  auto output_tensor = convolutional.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_3_4_1_1_0_1_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_3_4_1_1_0_1_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONVTestCPU, test_deconv_3_4_1_2_2_3_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(12));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(4));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(384));
  
  hypertea::DeconvolutionOp_CPU<float> convolutional = DeconvolutionOp_CPU<float>(weight.mutable_data(), bias.mutable_data(), 1, false, std::vector<int> {1,1}, std::vector<int> {2,2}, std::vector<int> {2,2}, std::vector<int> {3,3}, std::vector<int> {2,3,8,8}, std::vector<int> {2,4,11,11}, false);


  auto output_tensor = convolutional.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_3_4_1_2_2_3_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_3_4_1_2_2_3_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONVTestCPU, test_deconv_3_4_3_1_0_1_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(108));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(4));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(384));
  
  hypertea::DeconvolutionOp_CPU<float> convolutional = DeconvolutionOp_CPU<float>(weight.mutable_data(), bias.mutable_data(), 1, false, std::vector<int> {3,3}, std::vector<int> {1,1}, std::vector<int> {0,0}, std::vector<int> {1,1}, std::vector<int> {2,3,8,8}, std::vector<int> {2,4,10,10}, false);


  auto output_tensor = convolutional.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_3_4_3_1_0_1_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_3_4_3_1_0_1_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONVTestCPU, test_deconv_3_4_3_2_2_3_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(108));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(4));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(384));
  
  hypertea::DeconvolutionOp_CPU<float> convolutional = DeconvolutionOp_CPU<float>(weight.mutable_data(), bias.mutable_data(), 1, false, std::vector<int> {3,3}, std::vector<int> {2,2}, std::vector<int> {2,2}, std::vector<int> {3,3}, std::vector<int> {2,3,8,8}, std::vector<int> {2,4,17,17}, false);


  auto output_tensor = convolutional.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_3_4_3_2_2_3_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_3_4_3_2_2_3_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONVTestCPU, test_deconv_5_3_1_1_0_1_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(15));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(640));
  
  hypertea::DeconvolutionOp_CPU<float> convolutional = DeconvolutionOp_CPU<float>(weight.mutable_data(), bias.mutable_data(), 1, true, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {0,0}, std::vector<int> {1,1}, std::vector<int> {2,5,8,8}, std::vector<int> {2,3,8,8}, false);


  auto output_tensor = convolutional.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_5_3_1_1_0_1_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_5_3_1_1_0_1_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONVTestCPU, test_deconv_5_3_1_2_2_3_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(15));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(640));
  
  hypertea::DeconvolutionOp_CPU<float> convolutional = DeconvolutionOp_CPU<float>(weight.mutable_data(), bias.mutable_data(), 1, false, std::vector<int> {1,1}, std::vector<int> {2,2}, std::vector<int> {2,2}, std::vector<int> {3,3}, std::vector<int> {2,5,8,8}, std::vector<int> {2,3,11,11}, false);


  auto output_tensor = convolutional.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_5_3_1_2_2_3_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_5_3_1_2_2_3_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONVTestCPU, test_deconv_5_3_3_1_0_1_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(135));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(640));
  
  hypertea::DeconvolutionOp_CPU<float> convolutional = DeconvolutionOp_CPU<float>(weight.mutable_data(), bias.mutable_data(), 1, false, std::vector<int> {3,3}, std::vector<int> {1,1}, std::vector<int> {0,0}, std::vector<int> {1,1}, std::vector<int> {2,5,8,8}, std::vector<int> {2,3,10,10}, false);


  auto output_tensor = convolutional.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_5_3_3_1_0_1_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_5_3_3_1_0_1_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONVTestCPU, test_deconv_5_3_3_2_2_3_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(135));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(640));
  
  hypertea::DeconvolutionOp_CPU<float> convolutional = DeconvolutionOp_CPU<float>(weight.mutable_data(), bias.mutable_data(), 1, false, std::vector<int> {3,3}, std::vector<int> {2,2}, std::vector<int> {2,2}, std::vector<int> {3,3}, std::vector<int> {2,5,8,8}, std::vector<int> {2,3,17,17}, false);


  auto output_tensor = convolutional.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_5_3_3_2_2_3_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_5_3_3_2_2_3_result[i], 1e-3);
  }
}

}
