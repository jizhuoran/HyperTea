import torch.nn as nn
import torch
import itertools

import sys
sys.path.append('/home/zrji/hypertea_maker/pytorch')


from fake_random import FakeRandomGenerator
from oracle_util import *
from hypertea_generator.hypertea_generator import HyperteaGenerator

def conv_declare(module, input_data, output_data, op_name, opencl_collector):

    if module.bias is not None:
        bias_name = 'bias.mutable_data()'
    else:
        bias_name = 'NULL'


    weight_name = 'weight.mutable_data()'


    kernel_shape = list(module.kernel_size)
    stride = list(module.stride)
    padding = list(module.padding)
    dilation = list(module.dilation)


    is_1x1 = all(list(map(lambda x: x==1, kernel_shape + stride)) + list(map(lambda x: x==0, padding)))

    input_shape = list(input_data.shape)
    output_shape = list(output_data.shape)

    force_nd_conv = False

    conv_type = 'DeconvolutionOp' if module.transposed else 'ConvolutionOp'

    cpu_signature = '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(
                weight_name, bias_name, 1, bool2str_(is_1x1), 
                list2vecstr_(kernel_shape), list2vecstr_(stride), 
                list2vecstr_(padding), list2vecstr_(dilation),
                list2vecstr_(input_shape), list2vecstr_(output_shape),
                bool2str_(force_nd_conv))


    # libdnn = Libdnn(op_name+'_forward', groups,
    #                         conv_type, module.bias is not None, 
    #                         input_shape, output_shape, 
    #                         kernel_shape, padding, stride, dilation)

    # opencl_collector += libdnn.generate_libdnn_code()

    gpu_signature = cpu_signature

    # gpu_signature = '"{}_forward", {}, {}, {}, {}, {}'.format(
    #     op_name, prod_(output_shape), 
    #     weight_name, bias_name,
    #     list2vecstr_(libdnn.local_shape()),
    #     list2vecstr_(libdnn.global_shape()))

    return {'type':conv_type, 'op_name':op_name, 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature}



def generate_test_case(op_name, op_type, weight_shape, bias_shape, input_shape, cpu_signature):
    return f'''

TYPED_TEST(CONVTestCPU, test_{op_name}_CPU) {{
  typedef typename TypeParam::Dtype Dtype;
  
  fake_random_number random_generator;

  auto weight = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector({prod_(weight_shape)}));
  auto bias = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector({prod_(bias_shape)}));

  auto input_tensor = hypertea::TensorCPU<Dtype>(random_generator.generate_random_vector({prod_(input_shape)}));
  
  hypertea::{op_type}_CPU<float> convolutional = {op_type}_CPU<float>({cpu_signature});


  auto output_tensor = convolutional.Forward(input_tensor);

  auto output_data = output_tensor.debug_gtest_cpu_data();

  for (int i = 0; i < test_result::{op_name}_result.size(); ++i) {{
    EXPECT_NEAR(output_data[i], test_result::{op_name}_result[i], 1e-3);
  }}
}}

    '''



def conv_oracle_generator():

    conv_oracle = []
 

    in_out_channels = [(2, 3), (4, 3)]
    kernel_sizes = [1, 3]
    spd_sizes = [(1, 0, 1), (2, 2, 3)]
    
    conv_settings = list(itertools.product(in_out_channels, kernel_sizes, spd_sizes))

    for (in_channel, out_channel), kernel, (stride, padding, dilation) in conv_settings:

        rg = FakeRandomGenerator()

        conv = nn.Conv2d(in_channel, out_channel, kernel, stride=stride, padding=padding, dilation=dilation, bias=True)

        conv.weight.data = torch.tensor(rg.rn(conv.weight.shape), dtype = torch.float)
        conv.bias.data = torch.tensor(rg.rn(conv.bias.shape), dtype = torch.float)
        intput_tensor = torch.tensor(rg.rn((2, in_channel, 8, 8)), dtype = torch.float)

        conv_output = conv(intput_tensor)
        conv_result = '{ ' + str(conv_output.reshape(-1).tolist())[1:-1] + ' };'


        op_name = 'conv_{}_{}_{}_{}_{}_{}'.format(
            in_channel, out_channel, 
            kernel, stride, padding, dilation
        )

        conv_oracle.append(
            'std::vector<float> {}_result{}'.format(op_name,conv_result)
        )

        
        declare_info = conv_declare(conv, intput_tensor, conv_output, op_name, [])

        code = generate_test_case(
            op_name, declare_info['type'],
            list(conv.weight.shape), 
            list(conv.bias.shape), 
            list(intput_tensor.shape), 
            declare_info['cpu_signature']
        )

        print(code)

    save_oracle_result('conv', conv_oracle)        



def deconv_oracle_generator():

    deconv_oracle = []

    

    in_out_channels = [(2, 2), (3, 4), (5, 3)]
    kernel_sizes = [1, 3]

    spd_sizes = [(1, 0, 1), (2, 2, 3)]

    conv_settings = list(itertools.product(in_out_channels, kernel_sizes, spd_sizes))

    for (in_channel, out_channel), kernel, (stride, padding, dilation) in conv_settings:

        rg = FakeRandomGenerator()

        deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel, stride=stride, padding=padding, bias=True, dilation=dilation)

        deconv.weight.data = torch.tensor(rg.rn(deconv.weight.shape), dtype = torch.float)
        deconv.bias.data = torch.tensor(rg.rn(deconv.bias.shape), dtype = torch.float)
        intput_tensor = torch.tensor(rg.rn((2, in_channel, 8, 8)), dtype = torch.float)

        deconv_output = deconv(intput_tensor)

        deconv_result = '{ ' + str(deconv_output.reshape(-1).tolist())[1:-1] + ' };'

        op_name = 'deconv_{}_{}_{}_{}_{}_{}'.format(
            in_channel, out_channel, 
            kernel, stride, padding, dilation
        )
 
        deconv_oracle.append(
            'std::vector<float> {}_result{}'.format(
                op_name, deconv_result
            )
        )

        declare_info = conv_declare(deconv, intput_tensor, deconv_output, op_name, [])

        code = generate_test_case(
            op_name, declare_info['type'],
            list(deconv.weight.shape), 
            list(deconv.bias.shape), 
            list(intput_tensor.shape), 
            declare_info['cpu_signature']
        )

        print(code)

    save_oracle_result('deconv', deconv_oracle) 



def libdnn_conv_oracle_generator():

    conv_oracle = []
 

    in_out_channels = [(2, 3), (4, 3)]
    kernel_sizes = [1, 3]
    spd_sizes = [(1, 0, 1), (2, 2, 3)]
    
    conv_settings = list(itertools.product(in_out_channels, kernel_sizes, spd_sizes))

    for (in_channel, out_channel), kernel, (stride, padding, dilation) in conv_settings:

        rg = FakeRandomGenerator()

        conv = nn.Conv2d(in_channel, out_channel, kernel, stride=stride, padding=padding, dilation=dilation, bias=True)

        conv.weight.data = torch.tensor(rg.rn(conv.weight.shape), dtype = torch.float)
        conv.bias.data = torch.tensor(rg.rn(conv.bias.shape), dtype = torch.float)
        intput_tensor = torch.tensor(rg.rn((2, in_channel, 8, 8)), dtype = torch.float)

        conv_output = conv(intput_tensor)
        conv_result = '{ ' + str(conv_output.reshape(-1).tolist())[1:-1] + ' };'


        op_name = 'conv_{}_{}_{}_{}_{}_{}'.format(
            in_channel, out_channel, 
            kernel, stride, padding, dilation
        )

        conv_oracle.append(
            'std::vector<float> {}_result{}'.format(op_name,conv_result)
        )

        
        declare_info = conv_declare(conv, intput_tensor, conv_output, op_name, [])

        code = generate_test_case(
            op_name, declare_info['type'],
            list(conv.weight.shape), 
            list(conv.bias.shape), 
            list(intput_tensor.shape), 
            declare_info['cpu_signature']
        )
        
        libdnn = Libdnn(op_name+'_forward', module.groups,
                                conv_type, module.bias is not None, 
                                input_shape, output_shape, 
                                kernel_shape, padding, stride, dilation)

        opencl_collector += libdnn.generate_libdnn_code()

        
        print(code)

    save_oracle_result('conv', conv_oracle)


conv_oracle_generator()
deconv_oracle_generator()