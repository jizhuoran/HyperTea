#include <algorithm>
#include <vector>

#include "gtest/gtest.h"


#include "hypertea/common.hpp"

#include "test_hypertea_util.hpp"


namespace hypertea {


template <typename TypeParam>
class INPLACE_TENSOR_MATH_Test : public ::testing::Test {
 public:
  // using DeviceTensor = TypeParam;
 protected:
  INPLACE_TENSOR_MATH_Test() {
    hypertea::OpenCLHandler::Get().build_opencl_math_code(false);
  }
  virtual ~INPLACE_TENSOR_MATH_Test() {}
};


TYPED_TEST_CASE(INPLACE_TENSOR_MATH_Test, TestDtypes);


TYPED_TEST(INPLACE_TENSOR_MATH_Test, test_inplace_set) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  float alpha = 23.61;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  a.set(alpha);
  auto y_data = a.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], alpha, 1e-3);
  }

}



TYPED_TEST(INPLACE_TENSOR_MATH_Test, test_inplace_sqr) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();

  auto y = a.sqr();
  auto y_data = y.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], a_data.get()[i] * a_data.get()[i], 1e-3);
  }

}


TYPED_TEST(INPLACE_TENSOR_MATH_Test, test_inplace_sqrt) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N)).abs();
  auto a_data = a.debug_gtest_cpu_data();

  auto y = a.sqrt();
  auto y_data = y.debug_gtest_cpu_data();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], sqrt(a_data.get()[i]), 1e-3);
  }
}



TYPED_TEST(INPLACE_TENSOR_MATH_Test, test_inplace_powx) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N)).abs();
  auto a_data = a.debug_gtest_cpu_data();

  auto y = a.powx(1.5);
  auto y_data = y.debug_gtest_cpu_data();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], pow(a_data.get()[i], static_cast<float>(1.5)), 1e-3);
  }
}


TYPED_TEST(INPLACE_TENSOR_MATH_Test, test_inplace_exp) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();

  auto y = a.exp();
  auto y_data = y.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], exp(a_data.get()[i]), 1e-3);
  }
}


TYPED_TEST(INPLACE_TENSOR_MATH_Test, test_inplace_log) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N)).abs();
  auto a_data = a.debug_gtest_cpu_data();

  auto y = a.log();
  auto y_data = y.debug_gtest_cpu_data();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], log(a_data.get()[i]), 1e-3);
  }
}


TYPED_TEST(INPLACE_TENSOR_MATH_Test, test_inplace_abs) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();

  auto y = a.abs();
  auto y_data = y.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], std::abs(a_data.get()[i]), 1e-3);
  }
}


TYPED_TEST(INPLACE_TENSOR_MATH_Test, test_inplace_tanh) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();

  auto y = a.tanh();
  auto y_data = y.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], tanh(a_data.get()[i]), 1e-3);
  }
}

TYPED_TEST(INPLACE_TENSOR_MATH_Test, test_inplace_sigmoid) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();

  auto y = a.sigmoid();
  auto y_data = y.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], 0.5 * tanh(0.5 * a_data.get()[i]) + 0.5, 1e-3);
  }
}


TYPED_TEST(INPLACE_TENSOR_MATH_Test, test_inplace_elu) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  float alpha = 1.0;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();

  auto y = a.elu(alpha);
  auto y_data = y.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    auto truth_value = a_data.get()[i] > 0 ? a_data.get()[i] : alpha * (exp(a_data.get()[i]) - 1);
    EXPECT_NEAR(y_data.get()[i], truth_value, 1e-3);
  }
}


TYPED_TEST(INPLACE_TENSOR_MATH_Test, test_inplace_relu) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();

  float negative_slope = .0;

  auto y = a.relu(negative_slope);
  auto y_data = y.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    auto truth_value = a_data.get()[i] > 0 ? a_data.get()[i] : a_data.get()[i] * negative_slope;
    EXPECT_NEAR(y_data.get()[i], truth_value, 1e-3);
  }
}




TYPED_TEST(INPLACE_TENSOR_MATH_Test, test_inplace_add) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  auto b = DeviceTensor(random_generator.generate_random_vector(N));
  auto b_data = b.debug_gtest_cpu_data();

  a+=b;
  
  auto c_data = a.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] + b_data.get()[i], 1e-3);
  }
}


TYPED_TEST(INPLACE_TENSOR_MATH_Test, test_inplace_sub) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  auto b = DeviceTensor(random_generator.generate_random_vector(N));
  auto b_data = b.debug_gtest_cpu_data();

  a-=b;
  
  auto c_data = a.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] - b_data.get()[i], 1e-3);
  }
}

TYPED_TEST(INPLACE_TENSOR_MATH_Test, test_inplace_mul) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  auto b = DeviceTensor(random_generator.generate_random_vector(N));
  auto b_data = b.debug_gtest_cpu_data();

  a*=b;
  
  auto c_data = a.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] * b_data.get()[i], 1e-3);
  }
}

TYPED_TEST(INPLACE_TENSOR_MATH_Test, test_inplace_div) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  auto b = DeviceTensor(random_generator.generate_random_vector(N));
  auto b_data = b.debug_gtest_cpu_data();

  a/=b;
  
  auto c_data = a.debug_gtest_cpu_data();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] / b_data.get()[i], 1e-3);
  }
}





TYPED_TEST(INPLACE_TENSOR_MATH_Test, test_inplace_add_scalar) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  a+=61.23;
  
  auto c_data = a.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] + 61.23, 1e-3);
  }
}


TYPED_TEST(INPLACE_TENSOR_MATH_Test, test_inplace_sub_scalar) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  a-=61.23;
  
  auto c_data = a.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] - 61.23, 1e-3);
  }
}

TYPED_TEST(INPLACE_TENSOR_MATH_Test, test_inplace_mul_scalar) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  a*=61.23;
  
  auto c_data = a.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] * 61.23, 1e-3);
  }
}

TYPED_TEST(INPLACE_TENSOR_MATH_Test, test_inplace_div_scalar) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  a/=61.23;
  
  auto c_data = a.debug_gtest_cpu_data();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] / 61.23, 1e-3);
  }
}

}  // namespace caffe
