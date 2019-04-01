#include <algorithm>
#include <vector>

#include "gtest/gtest.h"


#include "hypertea/common.hpp"

#include "test_hypertea_util.hpp"
#include "hypertea/util/math_functions.hpp"


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

  const Dtype* a_data = a.cpu_data_gtest();
  const Dtype* y_data = y.cpu_data_gtest();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data[i], a_data[i] * a_data[i], 1e-3);
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

  const Dtype* a_data = a.cpu_data_gtest();
  const Dtype* y_data = y.cpu_data_gtest();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data[i], sqrt(a_data[i]), 1e-3);
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

  const Dtype* a_data = a.cpu_data_gtest();
  const Dtype* y_data = y.cpu_data_gtest();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data[i], pow(a_data[i], static_cast<float>(1.5)), 1e-3);
  }
}


TYPED_TEST(MATHFUNCTestCPU, test_exp_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto y = hypertea::TensorCPU<Dtype>(N);

  
  hypertea_exp(N, a.immutable_data(), y.mutable_data());

  const Dtype* a_data = a.cpu_data_gtest();
  const Dtype* y_data = y.cpu_data_gtest();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data[i], exp(a_data[i]), 1e-3);
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

  const Dtype* a_data = a.cpu_data_gtest();
  const Dtype* y_data = y.cpu_data_gtest();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data[i], log(a_data[i]), 1e-3);
  }
}


TYPED_TEST(MATHFUNCTestCPU, test_abs_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto y = hypertea::TensorCPU<Dtype>(N);

  
  hypertea_abs(N, a.immutable_data(), y.mutable_data());

  const Dtype* a_data = a.cpu_data_gtest();
  const Dtype* y_data = y.cpu_data_gtest();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data[i], std::abs(a_data[i]), 1e-3);
  }
}


TYPED_TEST(MATHFUNCTestCPU, test_tanh_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto y = hypertea::TensorCPU<Dtype>(N);

  
  hypertea_tanh(N, a.immutable_data(), y.mutable_data());

  const Dtype* a_data = a.cpu_data_gtest();
  const Dtype* y_data = y.cpu_data_gtest();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data[i], tanh(a_data[i]), 1e-3);
  }
}

TYPED_TEST(MATHFUNCTestCPU, test_sigmoid_CPU) {
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector(N));
  auto y = hypertea::TensorCPU<Dtype>(N);

  
  hypertea_sigmoid(N, a.immutable_data(), y.mutable_data());

  const Dtype* a_data = a.cpu_data_gtest();
  const Dtype* y_data = y.cpu_data_gtest();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data[i], 0.5 * tanh(0.5 * a_data[i]) + 0.5, 1e-3);
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

  const Dtype* a_data = a.cpu_data_gtest();
  const Dtype* b_data = b.cpu_data_gtest();
  const Dtype* c_data = c.cpu_data_gtest();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data[i], a_data[i] + b_data[i], 1e-3);
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

  const Dtype* a_data = a.cpu_data_gtest();
  const Dtype* b_data = b.cpu_data_gtest();
  const Dtype* c_data = c.cpu_data_gtest();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data[i], a_data[i] - b_data[i], 1e-3);
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

  const Dtype* a_data = a.cpu_data_gtest();
  const Dtype* b_data = b.cpu_data_gtest();
  const Dtype* c_data = c.cpu_data_gtest();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data[i], a_data[i] * b_data[i], 1e-3);
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

  const Dtype* a_data = a.cpu_data_gtest();
  const Dtype* b_data = b.cpu_data_gtest();
  const Dtype* c_data = c.cpu_data_gtest();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data[i], a_data[i] / b_data[i], 1e-3);
  }
}


}  // namespace caffe
