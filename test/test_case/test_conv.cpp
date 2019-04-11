#include <algorithm>
#include <vector>

#include "gtest/gtest.h"


#include "hypertea/common.hpp"

#include "test_hypertea_util.hpp"
#include "hypertea/operators/conv.hpp"
#include "hypertea/operators/deconv.hpp"
// #include "hypertea/operators/conv_op.hpp"

#include "test_result/conv_result.hpp"
#include "test_result/deconv_result.hpp"

namespace hypertea { 

template <typename TypeParam>
class CONV_Test : public ::testing::Test {
 public:
  // using DeviceTensor = TypeParam;
 protected:
  CONV_Test() {
    hypertea::OpenCLHandler::Get().build_opencl_math_code(false);
  }
  virtual ~CONV_Test() {}
};



TYPED_TEST_CASE(CONV_Test, TestDtypes);



TYPED_TEST(CONV_Test, test_conv_2_3_1_1_0_1) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  auto weight = DeviceTensor(random_generator.generate_random_vector(6));
  auto bias = DeviceTensor(random_generator.generate_random_vector(3));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(256));
  
  auto convolutional = ConvolutionOp<DeviceTensor>(&weight, &bias, 1, true, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {0,0}, std::vector<int> {1,1}, std::vector<int> {2,2,8,8}, std::vector<int> {2,3,8,8});


  auto output_tensor = convolutional(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::conv_2_3_1_1_0_1_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::conv_2_3_1_1_0_1_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONV_Test, test_conv_2_3_1_2_2_3) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  auto weight = DeviceTensor(random_generator.generate_random_vector(6));
  auto bias = DeviceTensor(random_generator.generate_random_vector(3));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(256));
  
  auto convolutional = ConvolutionOp<DeviceTensor>(&weight, &bias, 1, false, std::vector<int> {1,1}, std::vector<int> {2,2}, std::vector<int> {2,2}, std::vector<int> {3,3}, std::vector<int> {2,2,8,8}, std::vector<int> {2,3,6,6});


  auto output_tensor = convolutional(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::conv_2_3_1_2_2_3_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::conv_2_3_1_2_2_3_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONV_Test, test_conv_2_3_3_1_0_1) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  auto weight = DeviceTensor(random_generator.generate_random_vector(54));
  auto bias = DeviceTensor(random_generator.generate_random_vector(3));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(256));
  
  auto convolutional = ConvolutionOp<DeviceTensor>(&weight, &bias, 1, false, std::vector<int> {3,3}, std::vector<int> {1,1}, std::vector<int> {0,0}, std::vector<int> {1,1}, std::vector<int> {2,2,8,8}, std::vector<int> {2,3,6,6});


  auto output_tensor = convolutional(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::conv_2_3_3_1_0_1_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::conv_2_3_3_1_0_1_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONV_Test, test_conv_2_3_3_2_2_3) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  auto weight = DeviceTensor(random_generator.generate_random_vector(54));
  auto bias = DeviceTensor(random_generator.generate_random_vector(3));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(256));
  
  auto convolutional = ConvolutionOp<DeviceTensor>(&weight, &bias, 1, false, std::vector<int> {3,3}, std::vector<int> {2,2}, std::vector<int> {2,2}, std::vector<int> {3,3}, std::vector<int> {2,2,8,8}, std::vector<int> {2,3,3,3});


  auto output_tensor = convolutional(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::conv_2_3_3_2_2_3_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::conv_2_3_3_2_2_3_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONV_Test, test_conv_4_3_1_1_0_1) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  auto weight = DeviceTensor(random_generator.generate_random_vector(12));
  auto bias = DeviceTensor(random_generator.generate_random_vector(3));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(512));
  
  auto convolutional = ConvolutionOp<DeviceTensor>(&weight, &bias, 1, true, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {0,0}, std::vector<int> {1,1}, std::vector<int> {2,4,8,8}, std::vector<int> {2,3,8,8});


  auto output_tensor = convolutional(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::conv_4_3_1_1_0_1_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::conv_4_3_1_1_0_1_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONV_Test, test_conv_4_3_1_2_2_3) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  auto weight = DeviceTensor(random_generator.generate_random_vector(12));
  auto bias = DeviceTensor(random_generator.generate_random_vector(3));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(512));
  
  auto convolutional = ConvolutionOp<DeviceTensor>(&weight, &bias, 1, false, std::vector<int> {1,1}, std::vector<int> {2,2}, std::vector<int> {2,2}, std::vector<int> {3,3}, std::vector<int> {2,4,8,8}, std::vector<int> {2,3,6,6});


  auto output_tensor = convolutional(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::conv_4_3_1_2_2_3_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::conv_4_3_1_2_2_3_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONV_Test, test_conv_4_3_3_1_0_1) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  auto weight = DeviceTensor(random_generator.generate_random_vector(108));
  auto bias = DeviceTensor(random_generator.generate_random_vector(3));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(512));
  
  auto convolutional = ConvolutionOp<DeviceTensor>(&weight, &bias, 1, false, std::vector<int> {3,3}, std::vector<int> {1,1}, std::vector<int> {0,0}, std::vector<int> {1,1}, std::vector<int> {2,4,8,8}, std::vector<int> {2,3,6,6});


  auto output_tensor = convolutional(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::conv_4_3_3_1_0_1_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::conv_4_3_3_1_0_1_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONV_Test, test_conv_4_3_3_2_2_3) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  auto weight = DeviceTensor(random_generator.generate_random_vector(108));
  auto bias = DeviceTensor(random_generator.generate_random_vector(3));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(512));
  
  auto convolutional = ConvolutionOp<DeviceTensor>(&weight, &bias, 1, false, std::vector<int> {3,3}, std::vector<int> {2,2}, std::vector<int> {2,2}, std::vector<int> {3,3}, std::vector<int> {2,4,8,8}, std::vector<int> {2,3,3,3});


  auto output_tensor = convolutional(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::conv_4_3_3_2_2_3_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::conv_4_3_3_2_2_3_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONV_Test, test_deconv_2_2_1_1_0_1) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  auto weight = DeviceTensor(random_generator.generate_random_vector(4));
  auto bias = DeviceTensor(random_generator.generate_random_vector(2));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(256));
  
  auto convolutional = DeconvolutionOp<DeviceTensor>(&weight, &bias, 1, true, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {0,0}, std::vector<int> {1,1}, std::vector<int> {2,2,8,8}, std::vector<int> {2,2,8,8});


  auto output_tensor = convolutional(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_2_2_1_1_0_1_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_2_2_1_1_0_1_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONV_Test, test_deconv_2_2_1_2_2_3) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  auto weight = DeviceTensor(random_generator.generate_random_vector(4));
  auto bias = DeviceTensor(random_generator.generate_random_vector(2));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(256));
  
  auto convolutional = DeconvolutionOp<DeviceTensor>(&weight, &bias, 1, false, std::vector<int> {1,1}, std::vector<int> {2,2}, std::vector<int> {2,2}, std::vector<int> {3,3}, std::vector<int> {2,2,8,8}, std::vector<int> {2,2,11,11});


  auto output_tensor = convolutional(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_2_2_1_2_2_3_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_2_2_1_2_2_3_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONV_Test, test_deconv_2_2_3_1_0_1) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  auto weight = DeviceTensor(random_generator.generate_random_vector(36));
  auto bias = DeviceTensor(random_generator.generate_random_vector(2));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(256));
  
  auto convolutional = DeconvolutionOp<DeviceTensor>(&weight, &bias, 1, false, std::vector<int> {3,3}, std::vector<int> {1,1}, std::vector<int> {0,0}, std::vector<int> {1,1}, std::vector<int> {2,2,8,8}, std::vector<int> {2,2,10,10});


  auto output_tensor = convolutional(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_2_2_3_1_0_1_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_2_2_3_1_0_1_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONV_Test, test_deconv_2_2_3_2_2_3) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  auto weight = DeviceTensor(random_generator.generate_random_vector(36));
  auto bias = DeviceTensor(random_generator.generate_random_vector(2));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(256));
  
  auto convolutional = DeconvolutionOp<DeviceTensor>(&weight, &bias, 1, false, std::vector<int> {3,3}, std::vector<int> {2,2}, std::vector<int> {2,2}, std::vector<int> {3,3}, std::vector<int> {2,2,8,8}, std::vector<int> {2,2,17,17});


  auto output_tensor = convolutional(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_2_2_3_2_2_3_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_2_2_3_2_2_3_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONV_Test, test_deconv_3_4_1_1_0_1) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  auto weight = DeviceTensor(random_generator.generate_random_vector(12));
  auto bias = DeviceTensor(random_generator.generate_random_vector(4));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(384));
  
  auto convolutional = DeconvolutionOp<DeviceTensor>(&weight, &bias, 1, true, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {0,0}, std::vector<int> {1,1}, std::vector<int> {2,3,8,8}, std::vector<int> {2,4,8,8});


  auto output_tensor = convolutional(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_3_4_1_1_0_1_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_3_4_1_1_0_1_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONV_Test, test_deconv_3_4_1_2_2_3) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  auto weight = DeviceTensor(random_generator.generate_random_vector(12));
  auto bias = DeviceTensor(random_generator.generate_random_vector(4));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(384));
  
  auto convolutional = DeconvolutionOp<DeviceTensor>(&weight, &bias, 1, false, std::vector<int> {1,1}, std::vector<int> {2,2}, std::vector<int> {2,2}, std::vector<int> {3,3}, std::vector<int> {2,3,8,8}, std::vector<int> {2,4,11,11});


  auto output_tensor = convolutional(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_3_4_1_2_2_3_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_3_4_1_2_2_3_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONV_Test, test_deconv_3_4_3_1_0_1) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  auto weight = DeviceTensor(random_generator.generate_random_vector(108));
  auto bias = DeviceTensor(random_generator.generate_random_vector(4));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(384));
  
  auto convolutional = DeconvolutionOp<DeviceTensor>(&weight, &bias, 1, false, std::vector<int> {3,3}, std::vector<int> {1,1}, std::vector<int> {0,0}, std::vector<int> {1,1}, std::vector<int> {2,3,8,8}, std::vector<int> {2,4,10,10});


  auto output_tensor = convolutional(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_3_4_3_1_0_1_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_3_4_3_1_0_1_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONV_Test, test_deconv_3_4_3_2_2_3) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  auto weight = DeviceTensor(random_generator.generate_random_vector(108));
  auto bias = DeviceTensor(random_generator.generate_random_vector(4));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(384));
  
  auto convolutional = DeconvolutionOp<DeviceTensor>(&weight, &bias, 1, false, std::vector<int> {3,3}, std::vector<int> {2,2}, std::vector<int> {2,2}, std::vector<int> {3,3}, std::vector<int> {2,3,8,8}, std::vector<int> {2,4,17,17});


  auto output_tensor = convolutional(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_3_4_3_2_2_3_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_3_4_3_2_2_3_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONV_Test, test_deconv_5_3_1_1_0_1) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  auto weight = DeviceTensor(random_generator.generate_random_vector(15));
  auto bias = DeviceTensor(random_generator.generate_random_vector(3));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(640));
  
  auto convolutional = DeconvolutionOp<DeviceTensor>(&weight, &bias, 1, true, std::vector<int> {1,1}, std::vector<int> {1,1}, std::vector<int> {0,0}, std::vector<int> {1,1}, std::vector<int> {2,5,8,8}, std::vector<int> {2,3,8,8});


  auto output_tensor = convolutional(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_5_3_1_1_0_1_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_5_3_1_1_0_1_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONV_Test, test_deconv_5_3_1_2_2_3) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  auto weight = DeviceTensor(random_generator.generate_random_vector(15));
  auto bias = DeviceTensor(random_generator.generate_random_vector(3));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(640));
  
  auto convolutional = DeconvolutionOp<DeviceTensor>(&weight, &bias, 1, false, std::vector<int> {1,1}, std::vector<int> {2,2}, std::vector<int> {2,2}, std::vector<int> {3,3}, std::vector<int> {2,5,8,8}, std::vector<int> {2,3,11,11});


  auto output_tensor = convolutional(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_5_3_1_2_2_3_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_5_3_1_2_2_3_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONV_Test, test_deconv_5_3_3_1_0_1) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  auto weight = DeviceTensor(random_generator.generate_random_vector(135));
  auto bias = DeviceTensor(random_generator.generate_random_vector(3));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(640));
  
  auto convolutional = DeconvolutionOp<DeviceTensor>(&weight, &bias, 1, false, std::vector<int> {3,3}, std::vector<int> {1,1}, std::vector<int> {0,0}, std::vector<int> {1,1}, std::vector<int> {2,5,8,8}, std::vector<int> {2,3,10,10});


  auto output_tensor = convolutional(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_5_3_3_1_0_1_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_5_3_3_1_0_1_result[i], 1e-3);
  }
}

    


TYPED_TEST(CONV_Test, test_deconv_5_3_3_2_2_3) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  auto weight = DeviceTensor(random_generator.generate_random_vector(135));
  auto bias = DeviceTensor(random_generator.generate_random_vector(3));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(640));
  
  auto convolutional = DeconvolutionOp<DeviceTensor>(&weight, &bias, 1, false, std::vector<int> {3,3}, std::vector<int> {2,2}, std::vector<int> {2,2}, std::vector<int> {3,3}, std::vector<int> {2,5,8,8}, std::vector<int> {2,3,17,17});


  auto output_tensor = convolutional(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::deconv_5_3_3_2_2_3_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::deconv_5_3_3_2_2_3_result[i], 1e-3);
  }
}




}
