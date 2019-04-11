#include <algorithm>
#include <vector>

#include "gtest/gtest.h"


#include "hypertea/common.hpp"

#include "test_hypertea_util.hpp"


namespace hypertea {


template <typename TypeParam>
class MATHFUNCTestCPU : public ::testing::Test {
 public:
  typedef typename TypeParam::Dtype Dtype;
 protected:
  MATHFUNCTestCPU() {}
  virtual ~MATHFUNCTestCPU() {}
};

template <typename TypeParam>
class MATHFUNCTestGPU : public ::testing::Test {
 public:
  typedef typename TypeParam::Dtype Dtype;
 protected:
  MATHFUNCTestGPU() {
    hypertea::OpenCLHandler::Get().build_opencl_math_code(false);
  }
  virtual ~MATHFUNCTestGPU() {}
};


TYPED_TEST_CASE(MATHFUNCTestCPU, TestDtypesCPU);
TYPED_TEST_CASE(MATHFUNCTestGPU, TestDtypesGPU);


TYPED_TEST(MATHFUNCTestCPU, test_sqr_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto y = hypertea::TensorCPU<Dtype>(N);

  hypertea_sqr(N, a.immutable_data(), y.mutable_data());

  auto a_data = a.debug_gtest_cpu_data();
  auto y_data = y.debug_gtest_cpu_data();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], a_data.get()[i] * a_data.get()[i], 1e-3);
  }
}


TYPED_TEST(MATHFUNCTestCPU, test_sqrt_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto y = hypertea::TensorCPU<Dtype>(N);


  hypertea_sqr(N, a.immutable_data(), a.mutable_data());//make sure psedu random all postive

  hypertea_sqrt(N, a.immutable_data(), y.mutable_data());

  auto a_data = a.debug_gtest_cpu_data();
  auto y_data = y.debug_gtest_cpu_data();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], sqrt(a_data.get()[i]), 1e-3);
  }
}



TYPED_TEST(MATHFUNCTestCPU, test_powx_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto y = hypertea::TensorCPU<Dtype>(N);

  hypertea_sqr(N, a.immutable_data(), a.mutable_data());//make sure psedu random all postive

  hypertea_powx(N, a.immutable_data(), static_cast<float>(1.5), y.mutable_data());

  auto a_data = a.debug_gtest_cpu_data();
  auto y_data = y.debug_gtest_cpu_data();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], pow(a_data.get()[i], static_cast<float>(1.5)), 1e-3);
  }
}


TYPED_TEST(MATHFUNCTestCPU, test_exp_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto y = hypertea::TensorCPU<Dtype>(N);

  
  hypertea_exp(N, a.immutable_data(), y.mutable_data());

  auto a_data = a.debug_gtest_cpu_data();
  auto y_data = y.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], exp(a_data.get()[i]), 1e-3);
  }
}


TYPED_TEST(MATHFUNCTestCPU, test_log_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto y = hypertea::TensorCPU<Dtype>(N);

  hypertea_sqr(N, a.immutable_data(), a.mutable_data());//make sure psedu random all postive

  hypertea_log(N, a.immutable_data(), y.mutable_data());

  auto a_data = a.debug_gtest_cpu_data();
  auto y_data = y.debug_gtest_cpu_data();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], log(a_data.get()[i]), 1e-3);
  }
}


TYPED_TEST(MATHFUNCTestCPU, test_abs_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto y = hypertea::TensorCPU<Dtype>(N);

  
  hypertea_abs(N, a.immutable_data(), y.mutable_data());

  auto a_data = a.debug_gtest_cpu_data();
  auto y_data = y.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], std::abs(a_data.get()[i]), 1e-3);
  }
}


TYPED_TEST(MATHFUNCTestCPU, test_tanh_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto y = hypertea::TensorCPU<Dtype>(N);

  
  hypertea_tanh(N, a.immutable_data(), y.mutable_data());

  auto a_data = a.debug_gtest_cpu_data();
  auto y_data = y.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], tanh(a_data.get()[i]), 1e-3);
  }
}

TYPED_TEST(MATHFUNCTestCPU, test_sigmoid_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto y = hypertea::TensorCPU<Dtype>(N);

  
  hypertea_sigmoid(N, a.immutable_data(), y.mutable_data());

  auto a_data = a.debug_gtest_cpu_data();
  auto y_data = y.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], 0.5 * tanh(0.5 * a_data.get()[i]) + 0.5, 1e-3);
  }
}




TYPED_TEST(MATHFUNCTestCPU, test_add_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto b = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto c = hypertea::TensorCPU<Dtype>(N);

  hypertea_add(N, a.immutable_data(), b.immutable_data(), c.mutable_data());

  auto a_data = a.debug_gtest_cpu_data();
  auto b_data = b.debug_gtest_cpu_data();
  auto c_data = c.debug_gtest_cpu_data();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] + b_data.get()[i], 1e-3);
  }
}

TYPED_TEST(MATHFUNCTestCPU, test_sub_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto b = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto c = hypertea::TensorCPU<Dtype>(N);

  hypertea_sub(N, a.immutable_data(), b.immutable_data(), c.mutable_data());

  auto a_data = a.debug_gtest_cpu_data();
  auto b_data = b.debug_gtest_cpu_data();
  auto c_data = c.debug_gtest_cpu_data();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] - b_data.get()[i], 1e-3);
  }
}

TYPED_TEST(MATHFUNCTestCPU, test_mul_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto b = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto c = hypertea::TensorCPU<Dtype>(N);

  hypertea_mul(N, a.immutable_data(), b.immutable_data(), c.mutable_data());

  auto a_data = a.debug_gtest_cpu_data();
  auto b_data = b.debug_gtest_cpu_data();
  auto c_data = c.debug_gtest_cpu_data();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] * b_data.get()[i], 1e-3);
  }
}

TYPED_TEST(MATHFUNCTestCPU, test_div_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto b = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto c = hypertea::TensorCPU<Dtype>(N);

  hypertea_div(N, a.immutable_data(), b.immutable_data(), c.mutable_data());

  auto a_data = a.debug_gtest_cpu_data();
  auto b_data = b.debug_gtest_cpu_data();
  auto c_data = c.debug_gtest_cpu_data();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] / b_data.get()[i], 1e-3);
  }
}


}  // namespace caffe
