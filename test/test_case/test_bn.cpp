#include <algorithm>
#include <vector>

#include "gtest/gtest.h"


#include "hypertea/common.hpp"

#include "test_hypertea_util.hpp"
#include "hypertea/operators/batch_norm_op.hpp"
#include "test_result/bn_result.hpp"


namespace hypertea {

template <typename TypeParam>
class BNTest : public ::testing::Test {
 public:

 protected:
  BNTest() {
    hypertea::OpenCLHandler::Get().build_opencl_math_code(false);
  }
  virtual ~BNTest() {}
};


TYPED_TEST_CASE(BNTest, TestDtypes);

TYPED_TEST(BNTest, test_bn_1_t_f_f) {
  typedef TypeParam DeviceTensor;
  
  fake_random_number random_generator;

  
  
  auto weight = DeviceTensor(random_generator.generate_random_vector(1));
  auto bias = DeviceTensor(random_generator.generate_random_vector(1));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(32));
  
  auto bn = BatchNormOp<DeviceTensor>(1, 32, 1e-5, nullptr, nullptr, &weight, &bias, NOT_IN_PLACE);


  auto output_tensor = bn(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_t_f_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_t_f_f_result[i], 1e-3);
  }
}

    /*



TYPED_TEST(BNTest, test_bn_1_t_f_t) {
  typedef TypeParam DeviceTensor;
  
  fake_random_number random_generator;

  
  
  auto weight = DeviceTensor(random_generator.generate_random_vector(1));
  auto bias = DeviceTensor(random_generator.generate_random_vector(1));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(32));
  
  auto bn = BatchNormOp<DeviceTensor>(32, 2, 1, 1e-05, 1, false, TensorGPU<float>(0), TensorGPU<float>(0), &weight, &bias, IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_t_f_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_t_f_t_result[i], 1e-3);
  }
}
  


TYPED_TEST(BNTest, test_bn_1_t_t_f) {
  typedef TypeParam DeviceTensor;
  
  fake_random_number random_generator;

  auto mean = DeviceTensor(random_generator.generate_random_vector(1));
  auto var = DeviceTensor(random_generator.generate_random_vector(1));
  hypertea_abs(var.count(), &var, &var);
  auto weight = DeviceTensor(random_generator.generate_random_vector(1));
  auto bias = DeviceTensor(random_generator.generate_random_vector(1));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(32));
  
  auto bn = BatchNormOp<DeviceTensor>(32, 2, 1, 1e-05, 1, true, &mean, &var, &weight, &bias, NOT_IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_t_t_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_t_t_f_result[i], 1e-3);
  }
}

    

    


TYPED_TEST(BNTest, test_bn_1_t_t_t) {
  typedef TypeParam DeviceTensor;
  
  fake_random_number random_generator;

  auto mean = DeviceTensor(random_generator.generate_random_vector(1));
  auto var = DeviceTensor(random_generator.generate_random_vector(1));
  hypertea_abs(var.count(), &var, &var);
  auto weight = DeviceTensor(random_generator.generate_random_vector(1));
  auto bias = DeviceTensor(random_generator.generate_random_vector(1));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(32));
  
  auto bn = BatchNormOp<DeviceTensor>(32, 2, 1, 1e-05, 1, true, &mean, &var, &weight, &bias, IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_t_t_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_t_t_t_result[i], 1e-3);
  }
}

  

TYPED_TEST(BNTest, test_bn_1_f_f_f) {
  typedef TypeParam DeviceTensor;
  
  fake_random_number random_generator;

  
  
  
  

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(32));
  
  auto bn = BatchNormOp<DeviceTensor>(32, 2, 1, 1e-05, 1, false, TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), NOT_IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_f_f_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_f_f_f_result[i], 1e-3);
  }
}

    
  


TYPED_TEST(BNTest, test_bn_1_f_f_t) {
  typedef TypeParam DeviceTensor;
  
  fake_random_number random_generator;

  
  
  
  

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(32));
  
  auto bn = BatchNormOp<DeviceTensor>(32, 2, 1, 1e-05, 1, false, TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_f_f_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_f_f_t_result[i], 1e-3);
  }
}

   


TYPED_TEST(BNTest, test_bn_1_f_t_f) {
  typedef TypeParam DeviceTensor;
  
  fake_random_number random_generator;

  auto mean = DeviceTensor(random_generator.generate_random_vector(1));
  auto var = DeviceTensor(random_generator.generate_random_vector(1));
  hypertea_abs(var.count(), &var, &var);
  
  

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(32));
  
  auto bn = BatchNormOp<DeviceTensor>(32, 2, 1, 1e-05, 1, true, &mean, &var, TensorGPU<float>(0), TensorGPU<float>(0), NOT_IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_f_t_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_f_t_f_result[i], 1e-3);
  }
}

  


TYPED_TEST(BNTest, test_bn_1_f_t_t) {
  typedef TypeParam DeviceTensor;
  
  fake_random_number random_generator;

  auto mean = DeviceTensor(random_generator.generate_random_vector(1));
  auto var = DeviceTensor(random_generator.generate_random_vector(1));
  hypertea_abs(var.count(), &var, &var);
  
  

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(32));
  
  auto bn = BatchNormOp<DeviceTensor>(32, 2, 1, 1e-05, 1, true, &mean, &var, TensorGPU<float>(0), TensorGPU<float>(0), IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_1_f_t_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_1_f_t_t_result[i], 1e-3);
  }
}

    
  


TYPED_TEST(BNTest, test_bn_3_t_f_f) {
  typedef TypeParam DeviceTensor;
  
  fake_random_number random_generator;

  
  
  auto weight = DeviceTensor(random_generator.generate_random_vector(3));
  auto bias = DeviceTensor(random_generator.generate_random_vector(3));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(96));
  
  auto bn = BatchNormOp<DeviceTensor>(96, 2, 3, 1e-05, 1, false, TensorGPU<float>(0), TensorGPU<float>(0), &weight, &bias, NOT_IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_t_f_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_t_f_f_result[i], 1e-3);
  }
}

    



TYPED_TEST(BNTest, test_bn_3_t_f_t) {
  typedef TypeParam DeviceTensor;
  
  fake_random_number random_generator;

  
  
  auto weight = DeviceTensor(random_generator.generate_random_vector(3));
  auto bias = DeviceTensor(random_generator.generate_random_vector(3));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(96));
  
  auto bn = BatchNormOp<DeviceTensor>(96, 2, 3, 1e-05, 1, false, TensorGPU<float>(0), TensorGPU<float>(0), &weight, &bias, IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_t_f_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_t_f_t_result[i], 1e-3);
  }
}

    


    


TYPED_TEST(BNTest, test_bn_3_t_t_f) {
  typedef TypeParam DeviceTensor;
  
  fake_random_number random_generator;

  auto mean = DeviceTensor(random_generator.generate_random_vector(3));
  auto var = DeviceTensor(random_generator.generate_random_vector(3));
  hypertea_abs(var.count(), &var, &var);
  auto weight = DeviceTensor(random_generator.generate_random_vector(3));
  auto bias = DeviceTensor(random_generator.generate_random_vector(3));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(96));
  
  auto bn = BatchNormOp<DeviceTensor>(96, 2, 3, 1e-05, 1, true, &mean, &var, &weight, &bias, NOT_IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_t_t_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_t_t_f_result[i], 1e-3);
  }
}

    



TYPED_TEST(BNTest, test_bn_3_t_t_t) {
  typedef TypeParam DeviceTensor;
  
  fake_random_number random_generator;

  auto mean = DeviceTensor(random_generator.generate_random_vector(3));
  auto var = DeviceTensor(random_generator.generate_random_vector(3));
  hypertea_abs(var.count(), &var, &var);
  auto weight = DeviceTensor(random_generator.generate_random_vector(3));
  auto bias = DeviceTensor(random_generator.generate_random_vector(3));

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(96));
  
  auto bn = BatchNormOp<DeviceTensor>(96, 2, 3, 1e-05, 1, true, &mean, &var, &weight, &bias, IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_t_t_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_t_t_t_result[i], 1e-3);
  }
}

    



    


TYPED_TEST(BNTest, test_bn_3_f_f_f) {
  typedef TypeParam DeviceTensor;
  
  fake_random_number random_generator;

  
  
  
  

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(96));
  
  auto bn = BatchNormOp<DeviceTensor>(96, 2, 3, 1e-05, 1, false, TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), NOT_IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_f_f_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_f_f_f_result[i], 1e-3);
  }
}

    


    


TYPED_TEST(BNTest, test_bn_3_f_f_t) {
  typedef TypeParam DeviceTensor;
  
  fake_random_number random_generator;

  
  
  
  

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(96));
  
  auto bn = BatchNormOp<DeviceTensor>(96, 2, 3, 1e-05, 1, false, TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), TensorGPU<float>(0), IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_f_f_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_f_f_t_result[i], 1e-3);
  }
}

    
    


TYPED_TEST(BNTest, test_bn_3_f_t_f) {
  typedef TypeParam DeviceTensor;
  
  fake_random_number random_generator;

  auto mean = DeviceTensor(random_generator.generate_random_vector(3));
  auto var = DeviceTensor(random_generator.generate_random_vector(3));
  hypertea_abs(var.count(), &var, &var);
  
  

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(96));
  
  auto bn = BatchNormOp<DeviceTensor>(96, 2, 3, 1e-05, 1, true, &mean, &var, TensorGPU<float>(0), TensorGPU<float>(0), NOT_IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_f_t_f_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_f_t_f_result[i], 1e-3);
  }
}

    




TYPED_TEST(BNTest, test_bn_3_f_t_t) {
  typedef TypeParam DeviceTensor;
  
  fake_random_number random_generator;

  auto mean = DeviceTensor(random_generator.generate_random_vector(3));
  auto var = DeviceTensor(random_generator.generate_random_vector(3));
  hypertea_abs(var.count(), &var, &var);
  
  

  auto input_tensor = DeviceTensor(random_generator.generate_random_vector(96));
  
  auto bn = BatchNormOp<DeviceTensor>(96, 2, 3, 1e-05, 1, true, &mean, &var, TensorGPU<float>(0), TensorGPU<float>(0), IN_PLACE);


  auto output_tensor = bn.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::bn_3_f_t_t_result.size(); ++i) {
    EXPECT_NEAR(output_data.get()[i], test_result::bn_3_f_t_t_result[i], 1e-3);
  }
}



    */

    





}  // namespace caffe
