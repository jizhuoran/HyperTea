#include "hypertea/hypertea.hpp"

using DeviceTensor = hypertea::TensorGPU<float>;


int main(int argc, char const *argv[])
{

    std::vector<float> attn_mul_weight_vec{1, 2, 3, 4, 5, 6};
    std::vector<float> input_vec{1, 1, 10, 10};


    auto attn_mul_weight = DeviceTensor(attn_mul_weight_vec);
    auto input = DeviceTensor(input_vec);

    hypertea::LinearOp<DeviceTensor> attn_mul = hypertea::LinearOp<DeviceTensor> ( 
        &attn_mul_weight, nullptr, 2, 3 
    );


    auto output = attn_mul(input);

    auto output_data = output.debug_gtest_cpu_data();

    for (int i = 0; i < output.count(); ++i) {
        std::cout << output_data.get()[i] << " ";
    }
    std::cout << " " << std::endl;


    return 0;
}



