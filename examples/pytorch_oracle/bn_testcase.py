import torch.nn as nn
import torch
import itertools

from fake_random import FakeRandomGenerator
from oracle_util import *


def bn_declare(module, input_data, output_data, op_name, inplace, opencl_collector):



    top_shape = prod_(output_data.shape)
    num, channels = input_data.shape[:2]
    eps = module.eps
    scale_factor = 1
    use_global_stats = bool2str_(module.track_running_stats)


    if module.track_running_stats:
        mean_name, var_name = 'mean.mutable_data()', 'var.mutable_data()'
    else:
        mean_name = var_name = 'NULL'


    if module.affine:
        weight_name, bias_name = 'weight.mutable_data()', 'bias.mutable_data()'
    else:
        weight_name = bias_name = 'NULL'


    cpu_signature = '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(
                top_shape, num, channels, eps, scale_factor, 
                use_global_stats, mean_name, var_name,
                weight_name, bias_name, bool2inplace_str_(inplace))


    gpu_signature = '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(
                top_shape, num, channels, eps, scale_factor, 
                use_global_stats, mean_name, var_name,
                weight_name, bias_name, 1.0, 1.0, bool2inplace_str_(inplace))
    



    return {'type':'BatchNormOp', 'op_name':op_name, 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature}






def generate_cpu_test_case(op_name, weight_shape, bias_shape, mean_shape, var_shape, input_shape, cpu_signature):
    


    mean = 'auto mean = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector({}));'.format(prod_(mean_shape)) if mean_shape else ''
    var = 'auto var = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector({}));\n  hypertea_abs(var.count(), var.mutable_data(), var.mutable_data());'.format(prod_(var_shape)) if var_shape else ''
    weight = 'auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector({}));'.format(prod_(weight_shape)) if weight_shape else ''
    bias = 'auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector({}));'.format(prod_(bias_shape)) if bias_shape else ''


    return f'''

TYPED_TEST(BNTestCPU, test_{op_name}_CPU) {{
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  {mean}
  {var}
  {weight}
  {bias}

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector({prod_(input_shape)}));
  
  hypertea::BatchNormOp_CPU<float> bn = BatchNormOp_CPU<float>({cpu_signature});


  auto output_tensor = bn.Forward(input_tensor);

  const Dtype* output_data = output_tensor.cpu_data_gtest();

  for (int i = 0; i < test_result::{op_name}_result.size(); ++i) {{
    EXPECT_NEAR(output_data[i], test_result::{op_name}_result[i], 1e-3);
  }}
}}

    '''



def generate_gpu_test_case(op_name, weight_shape, bias_shape, mean_shape, var_shape, input_shape, gpu_signature):
    


    mean = 'auto mean = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector({}));'.format(prod_(mean_shape)) if mean_shape else ''
    var = 'auto var = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector({}));\n  hypertea_gpu_abs<Dtype>(var.count(), var.mutable_data(), var.mutable_data());'.format(prod_(var_shape)) if var_shape else ''
    weight = 'auto weight = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector({}));'.format(prod_(weight_shape)) if weight_shape else ''
    bias = 'auto bias = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector({}));'.format(prod_(bias_shape)) if bias_shape else ''


    return f'''

TYPED_TEST(BNTestGPU, test_{op_name}_GPU) {{
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  {mean}
  {var}
  {weight}
  {bias}

  auto input_tensor = hypertea::TensorGPU<Dtype>(random_generator.generate_random_vector({prod_(input_shape)}));
  
  hypertea::BatchNormOp_GPU<float> bn = BatchNormOp_GPU<float>({gpu_signature});


  auto output_tensor = bn.Forward(input_tensor);

  const Dtype* output_data = output_tensor.cpu_data_gtest();

  for (int i = 0; i < test_result::{op_name}_result.size(); ++i) {{
    EXPECT_NEAR(output_data[i], test_result::{op_name}_result[i], 1e-3);
  }}
}}

    '''



def bn_oracle_generator():

    def bool2signature_(x):
        return 't' if x else 'f'

    bn_oracle = []
 

    num_features = [1, 3]
    affines = [True, False]
    track_running_statses = [False, True]
    inplace = [False, True]
    
    bn_settings = list(itertools.product(num_features, affines, track_running_statses, inplace))

    for nf, af, tr, ip in bn_settings:

        rg = FakeRandomGenerator()

        bn = nn.BatchNorm2d(nf, affine = af, track_running_stats = tr).eval()

        # print(dir(bn))

        if tr:
            bn.running_mean.data = torch.tensor(rg.rn(bn.running_mean.shape), dtype = torch.float)
            bn.running_var.data = torch.abs(torch.tensor(rg.rn(bn.running_var.shape), dtype = torch.float))


        if af:
            bn.weight.data = torch.tensor(rg.rn(bn.weight.shape), dtype = torch.float)
            bn.bias.data = torch.tensor(rg.rn(bn.bias.shape), dtype = torch.float)



        intput_tensor = torch.tensor(rg.rn((2, nf, 4, 4)), dtype = torch.float)

        bn_output = bn(intput_tensor)
        bn_result = '{ ' + str(bn_output.reshape(-1).tolist())[1:-1] + ' };'


        op_name = 'bn_{}_{}_{}_{}'.format(nf, bool2signature_(af), bool2signature_(tr), bool2signature_(ip))

        bn_oracle.append(
            'std::vector<float> {}_result{}'.format(op_name,bn_result)
        )

        
        declare_info = bn_declare(bn, intput_tensor, bn_output, op_name, ip, [])

        # print(declare_info)


        code = generate_cpu_test_case(
            op_name,
            list(bn.weight.shape) if af else None, 
            list(bn.bias.shape) if af else None,
            list(bn.running_mean.shape) if tr else None,
            list(bn.running_var.shape) if tr else None,
            list(intput_tensor.shape), 
            declare_info['cpu_signature']
        )

        print(code)


        code = generate_gpu_test_case(
            op_name,
            list(bn.weight.shape) if af else None, 
            list(bn.bias.shape) if af else None,
            list(bn.running_mean.shape) if tr else None,
            list(bn.running_var.shape) if tr else None,
            list(intput_tensor.shape), 
            declare_info['gpu_signature']
        )

        print(code)
        
        # exit(0)

    save_oracle_result('bn', bn_oracle)        




bn_oracle_generator()
