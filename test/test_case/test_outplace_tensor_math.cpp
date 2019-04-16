#include <algorithm>
#include <vector>

#include "gtest/gtest.h"


#include "hypertea/common.hpp"

#include "test_hypertea_util.hpp"
// #include "hypertea/util/math_functions.hpp"


namespace hypertea {


template <typename TypeParam>
class OUTPLACE_TENSOR_MATH_Test : public ::testing::Test {
 public:
  // using DeviceTensor = TypeParam;
 protected:
  OUTPLACE_TENSOR_MATH_Test() {
#ifdef USE_OPENCL    
    hypertea::OpenCLHandler::Get().build_opencl_math_code(false);
#endif
  }
  virtual ~OUTPLACE_TENSOR_MATH_Test() {}
};



TYPED_TEST_CASE(OUTPLACE_TENSOR_MATH_Test, TestDtypes);


TYPED_TEST(OUTPLACE_TENSOR_MATH_Test, test_outplace_sqr) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();

  auto y = outplace_sqr(a);
  auto y_data = y.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], a_data.get()[i] * a_data.get()[i], 1e-3);
  }

}


TYPED_TEST(OUTPLACE_TENSOR_MATH_Test, test_outplace_sqrt) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  inplace_abs(a);
  auto a_data = a.debug_gtest_cpu_data();

  auto y = outplace_sqrt(a);
  auto y_data = y.debug_gtest_cpu_data();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], sqrt(a_data.get()[i]), 1e-3);
  }
}



TYPED_TEST(OUTPLACE_TENSOR_MATH_Test, test_outplace_powx) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  inplace_abs(a);
  auto a_data = a.debug_gtest_cpu_data();

  auto y = outplace_powx(a, 1.5);
  auto y_data = y.debug_gtest_cpu_data();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], pow(a_data.get()[i], static_cast<float>(1.5)), 1e-3);
  }
}


TYPED_TEST(OUTPLACE_TENSOR_MATH_Test, test_outplace_exp) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();

  auto y = outplace_exp(a);
  auto y_data = y.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], exp(a_data.get()[i]), 1e-3);
  }
}


TYPED_TEST(OUTPLACE_TENSOR_MATH_Test, test_outplace_log) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  inplace_abs(a);
  auto a_data = a.debug_gtest_cpu_data();

  auto y = outplace_log(a);
  auto y_data = y.debug_gtest_cpu_data();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], log(a_data.get()[i]), 1e-3);
  }
}


TYPED_TEST(OUTPLACE_TENSOR_MATH_Test, test_outplace_abs) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();

  auto y = outplace_abs(a);
  auto y_data = y.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], std::abs(a_data.get()[i]), 1e-3);
  }
}


TYPED_TEST(OUTPLACE_TENSOR_MATH_Test, test_outplace_tanh) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();

  auto y = outplace_tanh(a);
  auto y_data = y.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], tanh(a_data.get()[i]), 1e-3);
  }
}

TYPED_TEST(OUTPLACE_TENSOR_MATH_Test, test_outplace_sigmoid) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();

  auto y = outplace_sigmoid(a);
  auto y_data = y.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(y_data.get()[i], 0.5 * tanh(0.5 * a_data.get()[i]) + 0.5, 1e-3);
  }
}



TYPED_TEST(OUTPLACE_TENSOR_MATH_Test, test_outplace_elu) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();

  float alpha = 1.0;

  auto y = outplace_elu(a, alpha);
  auto y_data = y.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    auto truth_value = a_data.get()[i] > 0 ? a_data.get()[i] : alpha * (exp(a_data.get()[i]) - 1);
    EXPECT_NEAR(y_data.get()[i], truth_value, 1e-3);
  }
}

TYPED_TEST(OUTPLACE_TENSOR_MATH_Test, test_outplace_relu) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();

  float negative_slope = .0;


  auto y = outplace_relu(a, negative_slope);
  auto y_data = y.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    auto truth_value = a_data.get()[i] > 0 ? a_data.get()[i] : a_data.get()[i] * negative_slope;
    EXPECT_NEAR(y_data.get()[i], truth_value, 1e-3);
  }
}


TYPED_TEST(OUTPLACE_TENSOR_MATH_Test, test_outplace_add) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  auto b = DeviceTensor(random_generator.generate_random_vector(N));
  auto b_data = b.debug_gtest_cpu_data();

  auto c = a+b;
  auto c_data = c.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] + b_data.get()[i], 1e-3);
  }
}


TYPED_TEST(OUTPLACE_TENSOR_MATH_Test, test_outplace_sub) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  auto b = DeviceTensor(random_generator.generate_random_vector(N));
  auto b_data = b.debug_gtest_cpu_data();

  auto c = a-b;
  auto c_data = c.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] - b_data.get()[i], 1e-3);
  }
}

TYPED_TEST(OUTPLACE_TENSOR_MATH_Test, test_outplace_mul) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  auto b = DeviceTensor(random_generator.generate_random_vector(N));
  auto b_data = b.debug_gtest_cpu_data();

  auto c = a*b;
  auto c_data = c.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] * b_data.get()[i], 1e-3);
  }
}

TYPED_TEST(OUTPLACE_TENSOR_MATH_Test, test_outplace_div) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  auto b = DeviceTensor(random_generator.generate_random_vector(N));
  auto b_data = b.debug_gtest_cpu_data();

  auto c = a/b;
  auto c_data = c.debug_gtest_cpu_data();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] / b_data.get()[i], 1e-3);
  }
}





TYPED_TEST(OUTPLACE_TENSOR_MATH_Test, test_outplace_add_scalar) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  auto c = a+61.23;
  auto c_data = c.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] + 61.23, 1e-3);
  }
}


TYPED_TEST(OUTPLACE_TENSOR_MATH_Test, test_outplace_sub_scalar) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  auto c = a-61.23;
  auto c_data = c.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] - 61.23, 1e-3);
  }
}

TYPED_TEST(OUTPLACE_TENSOR_MATH_Test, test_outplace_mul_scalar) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  auto c = a*61.23;
  auto c_data = c.debug_gtest_cpu_data();

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] * 61.23, 1e-3);
  }
}

TYPED_TEST(OUTPLACE_TENSOR_MATH_Test, test_outplace_div_scalar) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 64;

  auto a = DeviceTensor(random_generator.generate_random_vector(N));
  auto a_data = a.debug_gtest_cpu_data();
  
  auto c = a/61.23;
  auto c_data = c.debug_gtest_cpu_data();


  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(c_data.get()[i], a_data.get()[i] / 61.23, 1e-3);
  }
}


TYPED_TEST(OUTPLACE_TENSOR_MATH_Test, test_outplace_avg_g128) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 2*64*63*32;

  auto a = DeviceTensor(random_generator.generate_random_vector(N)); //random_generator.generate_random_vector(N)
  // a.set(1);
  auto a_data = a.debug_gtest_cpu_data();
  

  auto mean = DeviceTensor(32);
  auto var = DeviceTensor(32);

  mean_var(a, mean, var, 32, 64*63, 1e-5);

  auto mean_data = mean.debug_gtest_cpu_data();
  auto var_data = var.debug_gtest_cpu_data();

  float mean_cpu[32];
  float var_cpu[32];


    /* code */
  for (int i = 0; i < 32; ++i) {
    mean_cpu[i] = 0;
    var_cpu[i] = 0;

    for (int batch_id = 0; batch_id < 2; ++batch_id) {
      for (int j = 0; j < 64*63; ++j) {
        mean_cpu[i] += a_data.get()[batch_id * 32*64*63 + i*64*63 + j];
        var_cpu[i] += (a_data.get()[batch_id * 32*64*63 + i*64*63 + j] * a_data.get()[batch_id * 32*64*63 + i*64*63 + j]);
      }
      
    }
    mean_cpu[i] /= (2*64*63);
    var_cpu[i] /= (2*64*63);
    var_cpu[i] -= (mean_cpu[i] * mean_cpu[i]);

    var_cpu[i] = sqrt(var_cpu[i] + 1e-05);
  }

  for (int i = 0; i < 32; ++i) {
    EXPECT_NEAR(mean_cpu[i], mean_data.get()[i], 1e-3);
  }

  for (int i = 0; i < 32; ++i) {
    EXPECT_NEAR(var_cpu[i], var_data.get()[i], 1e-3);
  }
}


TYPED_TEST(OUTPLACE_TENSOR_MATH_Test, test_outplace_avg_l128) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 2*64*32;

  auto a = DeviceTensor(random_generator.generate_random_vector(N)); //random_generator.generate_random_vector(N)
  // a.set(1);
  auto a_data = a.debug_gtest_cpu_data();
  

  auto mean = DeviceTensor(32);
  auto var = DeviceTensor(32);

  mean_var(a, mean, var, 32, 64, 1e-5);

  auto mean_data = mean.debug_gtest_cpu_data();
  auto var_data = var.debug_gtest_cpu_data();

  float mean_cpu[32];
  float var_cpu[32];


    /* code */
  for (int i = 0; i < 32; ++i) {
    mean_cpu[i] = 0;
    var_cpu[i] = 0;

    for (int batch_id = 0; batch_id < 2; ++batch_id) {
      for (int j = 0; j < 64; ++j) {
        mean_cpu[i] += a_data.get()[batch_id * 32*64 + i*64 + j];
        var_cpu[i] += (a_data.get()[batch_id * 32*64 + i*64 + j] * a_data.get()[batch_id * 32*64 + i*64 + j]);
      }
      
    }
    mean_cpu[i] /= (2*64);
    var_cpu[i] /= (2*64);
    var_cpu[i] -= (mean_cpu[i] * mean_cpu[i]);

    var_cpu[i] = sqrt(var_cpu[i] + 1e-05);
  }

  for (int i = 0; i < 32; ++i) {
    EXPECT_NEAR(mean_cpu[i], mean_data.get()[i], 1e-3);
  }

  for (int i = 0; i < 32; ++i) {
    EXPECT_NEAR(var_cpu[i], var_data.get()[i], 1e-3);
  }
}



TYPED_TEST(OUTPLACE_TENSOR_MATH_Test, test_outplace_argmax_l128) {
  
  using DeviceTensor = TypeParam;
  
  fake_random_number random_generator;
  const int N = 2*128;

  auto input_data = random_generator.generate_random_vector(N);

  auto a = DeviceTensor(input_data);

  auto max_index = batched_argmax(a, 128);

  int index_cpu[2];


  auto result = std::max_element(input_data.begin(), input_data.begin() + 128);
  index_cpu[0] = std::distance(input_data.begin(), result);

  result = std::max_element(input_data.begin() + 128, input_data.end());
  index_cpu[1] = std::distance(input_data.begin() + 128, result);


  for (int i = 0; i < 2; ++i) {
    EXPECT_NEAR(index_cpu[i], max_index[i], 1e-3);
  }
}



}  // namespace caffe
