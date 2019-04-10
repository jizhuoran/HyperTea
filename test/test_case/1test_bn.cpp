#include <algorithm>
#include <vector>

#include "gtest/gtest.h"


#include "hypertea/common.hpp"

#include "test_hypertea_util.hpp"
#include "hypertea/operators/batch_norm_op.hpp"
#include "test_result/bn_result.hpp"


namespace hypertea {


template <typename TypeParam>
class BNTestCPU : public ::testing::Test {
 public:
  typedef typename TypeParam::Dtype Dtype;
 protected:
  BNTestCPU() {}
  virtual ~BNTestCPU() {}
};

template <typename TypeParam>
class BNTestGPU : public ::testing::Test {
 public:
  typedef typename TypeParam::Dtype Dtype;
 protected:
  BNTestGPU() {
    hypertea::OpenCLHandler::Get().build_opencl_math_code(false);
  }
  virtual ~BNTestGPU() {}
};


TYPED_TEST_CASE(BNTestCPU, TestDtypesCPU);
TYPED_TEST_CASE(BNTestGPU, TestDtypesGPU);






TYPED_TEST(BNTestCPU, test_bn_1_t_f_f_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  
  
  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(1));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(1));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(32));
  
  hypertea::BatchNormOp_CPU<float> bn = BatchNormOp_CPU<float>(32, 2, 1, 1e-05, 1, false, TensorGPU<float>(0), TensorGPU<float>(0), weight, bias, NOT_IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_t_f_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_t_f_f_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestGPU, test_bn_1_t_f_f_GPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  
  
  auto weight = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(1));
  auto bias = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(1));

  auto input_tensor = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(32));
  
  hypertea::BatchNormOp_GPU<float> bn = BatchNormOp_GPU<float>(32, 2, 1, 1e-05, 1, false, TensorGPU<float>(0), TensorGPU<float>(0), weight, bias, 1.0, 1.0, NOT_IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_t_f_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_t_f_f_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestCPU, test_bn_1_t_f_t_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  
  
  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(1));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(1));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(32));
  
  hypertea::BatchNormOp_CPU<float> bn = BatchNormOp_CPU<float>(32, 2, 1, 1e-05, 1, false, TensorGPU<float>(0), TensorGPU<float>(0), weight, bias, IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_t_f_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_t_f_t_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestGPU, test_bn_1_t_f_t_GPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  
  
  auto weight = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(1));
  auto bias = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(1));

  auto input_tensor = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(32));
  
  hypertea::BatchNormOp_GPU<float> bn = BatchNormOp_GPU<float>(32, 2, 1, 1e-05, 1, false, TensorGPU<float>(0), TensorGPU<float>(0), weight, bias, 1.0, 1.0, IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_t_f_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_t_f_t_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestCPU, test_bn_1_t_t_f_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto mean = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(1));
  auto var = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(1));
  hypertea_abs(var.count(), var.mutable_data(), var.mutable_data());
  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(1));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(1));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(32));
  
  hypertea::BatchNormOp_CPU<float> bn = BatchNormOp_CPU<float>(32, 2, 1, 1e-05, 1, true, mean.mutable_data(), var.mutable_data(), weight, bias, NOT_IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_t_t_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_t_t_f_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestGPU, test_bn_1_t_t_f_GPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto mean = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(1));
  auto var = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(1));
  hypertea_gpu_abs<Dtype>(var.count(), var.mutable_data(), var.mutable_data());
  auto weight = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(1));
  auto bias = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(1));

  auto input_tensor = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(32));
  
  hypertea::BatchNormOp_GPU<float> bn = BatchNormOp_GPU<float>(32, 2, 1, 1e-05, 1, true, mean.mutable_data(), var.mutable_data(), weight, bias, 1.0, 1.0, NOT_IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_t_t_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_t_t_f_result[i], 1e-3);
  }
}

     


TYPED_TEST(BNTestCPU, test_bn_1_t_t_t_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto mean = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(1));
  auto var = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(1));
  hypertea_abs(var.count(), var.mutable_data(), var.mutable_data());
  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(1));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(1));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(32));
  
  hypertea::BatchNormOp_CPU<float> bn = BatchNormOp_CPU<float>(32, 2, 1, 1e-05, 1, true, mean.mutable_data(), var.mutable_data(), weight, bias, IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_t_t_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_t_t_t_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestGPU, test_bn_1_t_t_t_GPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto mean = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(1));
  auto var = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(1));
  hypertea_gpu_abs<Dtype>(var.count(), var.mutable_data(), var.mutable_data());
  auto weight = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(1));
  auto bias = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(1));

  auto input_tensor = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(32));
  
  hypertea::BatchNormOp_GPU<float> bn = BatchNormOp_GPU<float>(32, 2, 1, 1e-05, 1, true, mean.mutable_data(), var.mutable_data(), weight, bias, 1.0, 1.0, IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_t_t_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_t_t_t_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestCPU, test_bn_1_f_f_f_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  
  
  
  

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(32));
  
  hypertea::BatchNormOp_CPU<float> bn = BatchNormOp_CPU<float>(32, 2, 1, 1e-05, 1, false, TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), NOT_IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_f_f_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_f_f_f_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestGPU, test_bn_1_f_f_f_GPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  
  
  
  

  auto input_tensor = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(32));
  
  hypertea::BatchNormOp_GPU<float> bn = BatchNormOp_GPU<float>(32, 2, 1, 1e-05, 1, false, TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), 1.0, 1.0, NOT_IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_f_f_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_f_f_f_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestCPU, test_bn_1_f_f_t_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  
  
  
  

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(32));
  
  hypertea::BatchNormOp_CPU<float> bn = BatchNormOp_CPU<float>(32, 2, 1, 1e-05, 1, false, TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_f_f_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_f_f_t_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestGPU, test_bn_1_f_f_t_GPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  
  
  
  

  auto input_tensor = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(32));
  
  hypertea::BatchNormOp_GPU<float> bn = BatchNormOp_GPU<float>(32, 2, 1, 1e-05, 1, false, TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), 1.0, 1.0, IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_f_f_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_f_f_t_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestCPU, test_bn_1_f_t_f_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto mean = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(1));
  auto var = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(1));
  hypertea_abs(var.count(), var.mutable_data(), var.mutable_data());
  
  

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(32));
  
  hypertea::BatchNormOp_CPU<float> bn = BatchNormOp_CPU<float>(32, 2, 1, 1e-05, 1, true, mean.mutable_data(), var.mutable_data(), TensorGPU<float>(0), TensorGPU<float>(0), NOT_IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_f_t_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_f_t_f_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestGPU, test_bn_1_f_t_f_GPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto mean = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(1));
  auto var = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(1));
  hypertea_gpu_abs<Dtype>(var.count(), var.mutable_data(), var.mutable_data());
  
  

  auto input_tensor = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(32));
  
  hypertea::BatchNormOp_GPU<float> bn = BatchNormOp_GPU<float>(32, 2, 1, 1e-05, 1, true, mean.mutable_data(), var.mutable_data(), TensorGPU<float>(0), TensorGPU<float>(0), 1.0, 1.0, NOT_IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_f_t_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_f_t_f_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestCPU, test_bn_1_f_t_t_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto mean = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(1));
  auto var = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(1));
  hypertea_abs(var.count(), var.mutable_data(), var.mutable_data());
  
  

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(32));
  
  hypertea::BatchNormOp_CPU<float> bn = BatchNormOp_CPU<float>(32, 2, 1, 1e-05, 1, true, mean.mutable_data(), var.mutable_data(), TensorGPU<float>(0), TensorGPU<float>(0), IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_f_t_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_f_t_t_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestGPU, test_bn_1_f_t_t_GPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto mean = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(1));
  auto var = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(1));
  hypertea_gpu_abs<Dtype>(var.count(), var.mutable_data(), var.mutable_data());
  
  

  auto input_tensor = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(32));
  
  hypertea::BatchNormOp_GPU<float> bn = BatchNormOp_GPU<float>(32, 2, 1, 1e-05, 1, true, mean.mutable_data(), var.mutable_data(), TensorGPU<float>(0), TensorGPU<float>(0), 1.0, 1.0, IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_f_t_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_f_t_t_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestCPU, test_bn_3_t_f_f_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  
  
  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(96));
  
  hypertea::BatchNormOp_CPU<float> bn = BatchNormOp_CPU<float>(96, 2, 3, 1e-05, 1, false, TensorGPU<float>(0), TensorGPU<float>(0), weight, bias, NOT_IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_t_f_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_t_f_f_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestGPU, test_bn_3_t_f_f_GPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  
  
  auto weight = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(3));
  auto bias = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(3));

  auto input_tensor = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(96));
  
  hypertea::BatchNormOp_GPU<float> bn = BatchNormOp_GPU<float>(96, 2, 3, 1e-05, 1, false, TensorGPU<float>(0), TensorGPU<float>(0), weight, bias, 1.0, 1.0, NOT_IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_t_f_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_t_f_f_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestCPU, test_bn_3_t_f_t_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  
  
  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(96));
  
  hypertea::BatchNormOp_CPU<float> bn = BatchNormOp_CPU<float>(96, 2, 3, 1e-05, 1, false, TensorGPU<float>(0), TensorGPU<float>(0), weight, bias, IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_t_f_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_t_f_t_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestGPU, test_bn_3_t_f_t_GPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  
  
  auto weight = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(3));
  auto bias = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(3));

  auto input_tensor = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(96));
  
  hypertea::BatchNormOp_GPU<float> bn = BatchNormOp_GPU<float>(96, 2, 3, 1e-05, 1, false, TensorGPU<float>(0), TensorGPU<float>(0), weight, bias, 1.0, 1.0, IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_t_f_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_t_f_t_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestCPU, test_bn_3_t_t_f_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto mean = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));
  auto var = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));
  hypertea_abs(var.count(), var.mutable_data(), var.mutable_data());
  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(96));
  
  hypertea::BatchNormOp_CPU<float> bn = BatchNormOp_CPU<float>(96, 2, 3, 1e-05, 1, true, mean.mutable_data(), var.mutable_data(), weight, bias, NOT_IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_t_t_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_t_t_f_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestGPU, test_bn_3_t_t_f_GPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto mean = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(3));
  auto var = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(3));
  hypertea_gpu_abs<Dtype>(var.count(), var.mutable_data(), var.mutable_data());
  auto weight = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(3));
  auto bias = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(3));

  auto input_tensor = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(96));
  
  hypertea::BatchNormOp_GPU<float> bn = BatchNormOp_GPU<float>(96, 2, 3, 1e-05, 1, true, mean.mutable_data(), var.mutable_data(), weight, bias, 1.0, 1.0, NOT_IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_t_t_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_t_t_f_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestCPU, test_bn_3_t_t_t_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto mean = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));
  auto var = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));
  hypertea_abs(var.count(), var.mutable_data(), var.mutable_data());
  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(96));
  
  hypertea::BatchNormOp_CPU<float> bn = BatchNormOp_CPU<float>(96, 2, 3, 1e-05, 1, true, mean.mutable_data(), var.mutable_data(), weight, bias, IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_t_t_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_t_t_t_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestGPU, test_bn_3_t_t_t_GPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto mean = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(3));
  auto var = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(3));
  hypertea_gpu_abs<Dtype>(var.count(), var.mutable_data(), var.mutable_data());
  auto weight = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(3));
  auto bias = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(3));

  auto input_tensor = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(96));
  
  hypertea::BatchNormOp_GPU<float> bn = BatchNormOp_GPU<float>(96, 2, 3, 1e-05, 1, true, mean.mutable_data(), var.mutable_data(), weight, bias, 1.0, 1.0, IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_t_t_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_t_t_t_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestCPU, test_bn_3_f_f_f_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  
  
  
  

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(96));
  
  hypertea::BatchNormOp_CPU<float> bn = BatchNormOp_CPU<float>(96, 2, 3, 1e-05, 1, false, TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), NOT_IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_f_f_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_f_f_f_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestGPU, test_bn_3_f_f_f_GPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  
  
  
  

  auto input_tensor = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(96));
  
  hypertea::BatchNormOp_GPU<float> bn = BatchNormOp_GPU<float>(96, 2, 3, 1e-05, 1, false, TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), 1.0, 1.0, NOT_IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_f_f_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_f_f_f_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestCPU, test_bn_3_f_f_t_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  
  
  
  

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(96));
  
  hypertea::BatchNormOp_CPU<float> bn = BatchNormOp_CPU<float>(96, 2, 3, 1e-05, 1, false, TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_f_f_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_f_f_t_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestGPU, test_bn_3_f_f_t_GPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  
  
  
  

  auto input_tensor = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(96));
  
  hypertea::BatchNormOp_GPU<float> bn = BatchNormOp_GPU<float>(96, 2, 3, 1e-05, 1, false, TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), 1.0, 1.0, IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_f_f_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_f_f_t_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestCPU, test_bn_3_f_t_f_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto mean = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));
  auto var = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));
  hypertea_abs(var.count(), var.mutable_data(), var.mutable_data());
  
  

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(96));
  
  hypertea::BatchNormOp_CPU<float> bn = BatchNormOp_CPU<float>(96, 2, 3, 1e-05, 1, true, mean.mutable_data(), var.mutable_data(), TensorGPU<float>(0), TensorGPU<float>(0), NOT_IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_f_t_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_f_t_f_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestGPU, test_bn_3_f_t_f_GPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto mean = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(3));
  auto var = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(3));
  hypertea_gpu_abs<Dtype>(var.count(), var.mutable_data(), var.mutable_data());
  
  

  auto input_tensor = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(96));
  
  hypertea::BatchNormOp_GPU<float> bn = BatchNormOp_GPU<float>(96, 2, 3, 1e-05, 1, true, mean.mutable_data(), var.mutable_data(), TensorGPU<float>(0), TensorGPU<float>(0), 1.0, 1.0, NOT_IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_f_t_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_f_t_f_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestCPU, test_bn_3_f_t_t_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto mean = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));
  auto var = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(3));
  hypertea_abs(var.count(), var.mutable_data(), var.mutable_data());
  
  

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(96));
  
  hypertea::BatchNormOp_CPU<float> bn = BatchNormOp_CPU<float>(96, 2, 3, 1e-05, 1, true, mean.mutable_data(), var.mutable_data(), TensorGPU<float>(0), TensorGPU<float>(0), IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_f_t_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_f_t_t_result[i], 1e-3);
  }
}

    


TYPED_TEST(BNTestGPU, test_bn_3_f_t_t_GPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto mean = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(3));
  auto var = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(3));
  hypertea_gpu_abs<Dtype>(var.count(), var.mutable_data(), var.mutable_data());
  
  

  auto input_tensor = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector(96));
  
  hypertea::BatchNormOp_GPU<float> bn = BatchNormOp_GPU<float>(96, 2, 3, 1e-05, 1, true, mean.mutable_data(), var.mutable_data(), TensorGPU<float>(0), TensorGPU<float>(0), 1.0, 1.0, IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_f_t_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_f_t_t_result[i], 1e-3);
  }
}

    

    





}  // namespace caffe
