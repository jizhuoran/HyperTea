#include <algorithm>
#include <vector>

#include "gtest/gtest.h"


#include "hypertea/common.hpp"

#include "test_hypertea_util.hpp"
#include "hypertea/operators/activation.hpp"


namespace hypertea {


template <typename TypeParam>
class ACTIVATION_Test : public ::testing::Test {
 public:
  // typedef typename TypeParam::Dtype Dtype;
 protected:
  ACTIVATION_Test() {
    hypertea::OpenCLHandler::Get().build_opencl_math_code(false);
  }
  virtual ~ACTIVATION_Test() {}
};



TYPED_TEST_CASE(ACTIVATION_Test, TestDtypes);


TYPED_TEST(ACTIVATION_Test, test_inplace_relu) {

  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  float negative_slope_ = 0.1;

  auto relu_op = ReLUOp<DeviceTensor>(negative_slope_, true);

  auto a = DeviceTensor(random_generator.generate_random_vector(64));
  auto a_data = a.debug_gtest_cpu_data();
  
  relu_op(a);
  
  auto y_data = a.debug_gtest_cpu_data();

  for (int i = 0; i < a.count(); ++i) {
    auto ground_truth = std::max(a_data.get()[i], float(0)) + negative_slope_ * std::min(a_data.get()[i], float(0));
    EXPECT_NEAR(y_data.get()[i], ground_truth, 1e-3);
  }

}


TYPED_TEST(ACTIVATION_Test, test_outplace_relu) {

  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  float negative_slope_ = 0.1;

  auto relu_op = ReLUOp<DeviceTensor>(negative_slope_, false);

  auto a = DeviceTensor(random_generator.generate_random_vector(64));
  auto a_data = a.debug_gtest_cpu_data();
  
  auto y = relu_op(a);
  
  auto newa_data = a.debug_gtest_cpu_data();
  auto y_data = y.debug_gtest_cpu_data();

  for (int i = 0; i < a.count(); ++i) {
    auto ground_truth = std::max(a_data.get()[i], float(0)) + negative_slope_ * std::min(a_data.get()[i], float(0));
    EXPECT_NEAR(y_data.get()[i], ground_truth, 1e-3);
    EXPECT_NEAR(a_data.get()[i], newa_data.get()[i], 1e-3);
  }

}





TYPED_TEST(ACTIVATION_Test, test_inplace_elu) {

  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  float alpha = 0.9;

  auto elu_op = ELUOp<DeviceTensor>(alpha, true);

  auto a = DeviceTensor(random_generator.generate_random_vector(64));
  auto a_data = a.debug_gtest_cpu_data();
  
  elu_op(a);
  
  auto y_data = a.debug_gtest_cpu_data();

  for (int i = 0; i < a.count(); ++i) {
    auto ground_truth = std::max(a_data.get()[i], float(0)) + alpha * (exp(std::min(a_data.get()[i], float(0))) - float(1));
    EXPECT_NEAR(y_data.get()[i], ground_truth, 1e-3);
  }

}


TYPED_TEST(ACTIVATION_Test, test_outplace_elu) {

  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;

  float alpha = 0.9;

  auto elu_op = ELUOp<DeviceTensor>(alpha, false);

  auto a = DeviceTensor(random_generator.generate_random_vector(64));
  auto a_data = a.debug_gtest_cpu_data();
  
  auto y = elu_op(a);
  
  auto newa_data = a.debug_gtest_cpu_data();
  auto y_data = y.debug_gtest_cpu_data();

  for (int i = 0; i < a.count(); ++i) {
    auto ground_truth = std::max(a_data.get()[i], float(0)) + alpha * (exp(std::min(a_data.get()[i], float(0))) - float(1));
    EXPECT_NEAR(y_data.get()[i], ground_truth, 1e-3);
    EXPECT_NEAR(a_data.get()[i], newa_data.get()[i], 1e-3);
  }

}




TYPED_TEST(ACTIVATION_Test, test_inplace_tanh) {

  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;


  auto tanh_op = TanHOp<DeviceTensor>(true);

  auto a = DeviceTensor(random_generator.generate_random_vector(64));
  auto a_data = a.debug_gtest_cpu_data();
  
  tanh_op(a);
  
  auto y_data = a.debug_gtest_cpu_data();

  for (int i = 0; i < a.count(); ++i) {
    EXPECT_NEAR(y_data.get()[i], tanh(a_data.get()[i]), 1e-3);
  }

}


TYPED_TEST(ACTIVATION_Test, test_outplace_tanh) {

  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;


  auto tanh_op = TanHOp<DeviceTensor>(false);

  auto a = DeviceTensor(random_generator.generate_random_vector(64));
  auto a_data = a.debug_gtest_cpu_data();
  
  auto y = tanh_op(a);
  
  auto newa_data = a.debug_gtest_cpu_data();
  auto y_data = y.debug_gtest_cpu_data();

  for (int i = 0; i < a.count(); ++i) {
    EXPECT_NEAR(y_data.get()[i], tanh(a_data.get()[i]), 1e-3);
    EXPECT_NEAR(a_data.get()[i], newa_data.get()[i], 1e-3);
  }

}

}  // namespace caffe
