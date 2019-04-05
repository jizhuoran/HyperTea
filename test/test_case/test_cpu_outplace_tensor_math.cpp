#include <algorithm>
#include <vector>

#include "gtest/gtest.h"


#include "hypertea/common.hpp"

#include "test_hypertea_util.hpp"
// #include "hypertea/util/math_functions.hpp"


namespace hypertea {


template <typename TypeParam>
class OUTPLACE_TENSOR_MATH_TestCPU : public ::testing::Test {
 public:
  typedef typename TypeParam::Dtype Dtype;
 protected:
  OUTPLACE_TENSOR_MATH_TestCPU() {}
  virtual ~OUTPLACE_TENSOR_MATH_TestCPU() {}
};

TYPED_TEST_CASE(OUTPLACE_TENSOR_MATH_TestCPU, TestDtypesCPU);


// TYPED_TEST(OUTPLACE_TENSOR_MATH_TestCPU, test_outplace_sqr_CPU) {
//   typedef typename TypeParam::Dtype Dtype;
  
//   fake_random_number random_generator;
//   const int N = 64;

//   auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
//   auto a_data = a.debug_gtest_cpu_data();

//   auto y = cpu_sqr(a);
//   auto y_data = y.debug_gtest_cpu_data();

//   for (int i = 0; i < N; ++i) {
//     EXPECT_NEAR(y_data.get()[i], a_data.get()[i] * a_data.get()[i], 1e-3);
//   }

// }


// TYPED_TEST(OUTPLACE_TENSOR_MATH_TestCPU, test_outplace_sqrt_CPU) {
//   typedef typename TypeParam::Dtype Dtype;
  
//   fake_random_number random_generator;
//   const int N = 64;

//   auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N)).abs();
//   auto a_data = a.debug_gtest_cpu_data();

//   auto y = cpu_sqrt(a);
//   auto y_data = y.debug_gtest_cpu_data();


//   for (int i = 0; i < N; ++i) {
//     EXPECT_NEAR(y_data.get()[i], sqrt(a_data.get()[i]), 1e-3);
//   }
// }



// TYPED_TEST(OUTPLACE_TENSOR_MATH_TestCPU, test_outplace_powx_CPU) {
//   typedef typename TypeParam::Dtype Dtype;
  
//   fake_random_number random_generator;
//   const int N = 64;

//   auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N)).abs();
//   auto a_data = a.debug_gtest_cpu_data();

//   auto y = cpu_powx(a, 1.5);
//   auto y_data = y.debug_gtest_cpu_data();


//   for (int i = 0; i < N; ++i) {
//     EXPECT_NEAR(y_data.get()[i], pow(a_data.get()[i], static_cast<float>(1.5)), 1e-3);
//   }
// }


// TYPED_TEST(OUTPLACE_TENSOR_MATH_TestCPU, test_outplace_exp_CPU) {
//   typedef typename TypeParam::Dtype Dtype;
  
//   fake_random_number random_generator;
//   const int N = 64;

//   auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
//   auto a_data = a.debug_gtest_cpu_data();

//   auto y = cpu_exp(a);
//   auto y_data = y.debug_gtest_cpu_data();

//   for (int i = 0; i < N; ++i) {
//     EXPECT_NEAR(y_data.get()[i], exp(a_data.get()[i]), 1e-3);
//   }
// }


TYPED_TEST(OUTPLACE_TENSOR_MATH_TestCPU, test_outplace_log_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N)).abs();
  auto a_data = a.debug_gtest_cpu_data();

  auto y = cpu_log(a);
  auto y_data = y.debug_gtest_cpu_data();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], log(a_data.get()[i]), 1e-3);
  }
}


// TYPED_TEST(OUTPLACE_TENSOR_MATH_TestCPU, test_outplace_abs_CPU) {
//   typedef typename TypeParam::Dtype Dtype;
  
//   fake_random_number random_generator;
//   const int N = 64;

//   auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
//   auto a_data = a.debug_gtest_cpu_data();

//   auto y = cpu_abs(a);
//   auto y_data = y.debug_gtest_cpu_data();

//   for (int i = 0; i < N; ++i) {
//     EXPECT_NEAR(y_data.get()[i], std::abs(a_data.get()[i]), 1e-3);
//   }
// }


// TYPED_TEST(OUTPLACE_TENSOR_MATH_TestCPU, test_outplace_tanh_CPU) {
//   typedef typename TypeParam::Dtype Dtype;
  
//   fake_random_number random_generator;
//   const int N = 64;

//   auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
//   auto a_data = a.debug_gtest_cpu_data();

//   auto y = cpu_tanh(a);
//   auto y_data = y.debug_gtest_cpu_data();

//   for (int i = 0; i < N; ++i) {
//     EXPECT_NEAR(y_data.get()[i], tanh(a_data.get()[i]), 1e-3);
//   }
// }

// TYPED_TEST(OUTPLACE_TENSOR_MATH_TestCPU, test_outplace_sigmoid_CPU) {
//   typedef typename TypeParam::Dtype Dtype;
  
//   fake_random_number random_generator;
//   const int N = 64;

//   auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
//   auto a_data = a.debug_gtest_cpu_data();

//   auto y = cpu_sigmoid(a);
//   auto y_data = y.debug_gtest_cpu_data();

//   for (int i = 0; i < N; ++i) {
//     EXPECT_NEAR(y_data.get()[i], 0.5 * tanh(0.5 * a_data.get()[i]) + 0.5, 1e-3);
//   }
// }



// TYPED_TEST(OUTPLACE_TENSOR_MATH_TestCPU, test_outplace_elu_CPU) {
//   typedef typename TypeParam::Dtype Dtype;
  
//   fake_random_number random_generator;
//   const int N = 64;

//   auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
//   auto a_data = a.debug_gtest_cpu_data();

//   float alpha = 1.0;

//   auto y = cpu_elu(a, alpha);
//   auto y_data = y.debug_gtest_cpu_data();

//   for (int i = 0; i < N; ++i) {
//     auto truth_value = a_data.get()[i] > 0 ? a_data.get()[i] : alpha * (exp(a_data.get()[i]) - 1);
//     EXPECT_NEAR(y_data.get()[i], truth_value, 1e-3);
//   }
// }

// TYPED_TEST(OUTPLACE_TENSOR_MATH_TestCPU, test_outplace_relu_CPU) {
//   typedef typename TypeParam::Dtype Dtype;
  
//   fake_random_number random_generator;
//   const int N = 64;

//   auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
//   auto a_data = a.debug_gtest_cpu_data();

//   float negative_slope = .0;


//   auto y = cpu_relu(a, negative_slope);
//   auto y_data = y.debug_gtest_cpu_data();

//   for (int i = 0; i < N; ++i) {
//     auto truth_value = a_data.get()[i] > 0 ? a_data.get()[i] : a_data.get()[i] * negative_slope;
//     EXPECT_NEAR(y_data.get()[i], truth_value, 1e-3);
//   }
// }


TYPED_TEST(OUTPLACE_TENSOR_MATH_TestCPU, test_outplace_add_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  auto b = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto b_data = b.debug_gtest_cpu_data();

  auto c = a+b;
  auto c_data = c.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] + b_data.get()[i], 1e-3);
  }
}


TYPED_TEST(OUTPLACE_TENSOR_MATH_TestCPU, test_outplace_sub_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  auto b = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto b_data = b.debug_gtest_cpu_data();

  auto c = a-b;
  auto c_data = c.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] - b_data.get()[i], 1e-3);
  }
}

TYPED_TEST(OUTPLACE_TENSOR_MATH_TestCPU, test_outplace_mul_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  auto b = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto b_data = b.debug_gtest_cpu_data();

  auto c = a*b;
  auto c_data = c.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] * b_data.get()[i], 1e-3);
  }
}

TYPED_TEST(OUTPLACE_TENSOR_MATH_TestCPU, test_outplace_div_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  auto b = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto b_data = b.debug_gtest_cpu_data();

  auto c = a/b;
  auto c_data = c.debug_gtest_cpu_data();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] / b_data.get()[i], 1e-3);
  }
}





TYPED_TEST(OUTPLACE_TENSOR_MATH_TestCPU, test_outplace_add_scalar_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  auto c = a+61.23;
  auto c_data = c.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] + 61.23, 1e-3);
  }
}


TYPED_TEST(OUTPLACE_TENSOR_MATH_TestCPU, test_outplace_sub_scalar_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  auto c = a-61.23;
  auto c_data = c.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] - 61.23, 1e-3);
  }
}

TYPED_TEST(OUTPLACE_TENSOR_MATH_TestCPU, test_outplace_mul_scalar_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  auto c = a*61.23;
  auto c_data = c.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] * 61.23, 1e-3);
  }
}

TYPED_TEST(OUTPLACE_TENSOR_MATH_TestCPU, test_outplace_div_scalar_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  auto c = a/61.23;
  auto c_data = c.debug_gtest_cpu_data();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] / 61.23, 1e-3);
  }
}


}  // namespace caffe
