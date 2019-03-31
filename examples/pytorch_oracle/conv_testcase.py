import torch.nn as nn
import torch
import itertools

from fake_random import FakeRandomGenerator
from oracle_util import save_oracle_result


def conv_oracle_generator():

    conv_oracle = []

    rg = FakeRandomGenerator()

    in_out_channels = [(4, 4), (3, 6), (8, 3)]
    kernel_sizes = [1, 4]
    stride_sizes = [1, 4]
    padding_sizes = [0, 4]
    dilation_sizes = [1, 4]

    conv_settings = list(itertools.product(in_out_channels, kernel_sizes, stride_sizes, padding_sizes, dilation_sizes))

    for (in_channel, out_channel), kernel, stride, padding, dilation in conv_settings:

        conv = nn.Conv2d(in_channel, out_channel, kernel, stride=stride, padding=padding, dilation=dilation, bias=True)

        conv.weight.data = torch.tensor(rg.rn(conv.weight.shape), dtype = torch.float)
        conv.bias.data = torch.tensor(rg.rn(conv.bias.shape), dtype = torch.float)
        intput_tensor = torch.tensor(rg.rn((2, in_channel, 32, 32)), dtype = torch.float)

        conv_output = conv(intput_tensor)
        conv_result = '{ ' + str(conv_output.reshape(-1).tolist())[1:-1] + ' };'

        conv_oracle.append(
            'std::vector<float> conv_{}_{}_{}_{}_{}_{}_result{}'.format(
                in_channel, out_channel, 
                kernel, stride, padding, dilation,
                conv_result
            )
        )

    save_oracle_result('conv', conv_oracle)        
    # save_oracle_result('conv', conv_oracle)        



def deconv_oracle_generator():

    deconv_oracle = []

    rg = FakeRandomGenerator()

    in_out_channels = [(4, 4), (3, 6), (8, 3)]
    kernel_sizes = [1, 4]
    stride_sizes = [1, 4]
    padding_sizes = [0, 4]
    dilation_sizes = [1, 4]

    conv_settings = list(itertools.product(in_out_channels, kernel_sizes, stride_sizes, padding_sizes, dilation_sizes))

    for (in_channel, out_channel), kernel, stride, padding, dilation in conv_settings:

        deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel, stride=stride, padding=padding, bias=True, dilation=dilation)

        deconv.weight.data = torch.tensor(rg.rn(deconv.weight.shape), dtype = torch.float)
        deconv.bias.data = torch.tensor(rg.rn(deconv.bias.shape), dtype = torch.float)
        intput_tensor = torch.tensor(rg.rn((2, in_channel, 32, 32)), dtype = torch.float)

        deconv_output = deconv(intput_tensor)

        deconv_result = '{ ' + str(deconv_output.reshape(-1).tolist())[1:-1] + ' };'

        deconv_oracle.append(
            'std::vector<float> deconv_{}_{}_{}_{}_{}_{}_result{}'.format(
                in_channel, out_channel, 
                kernel, stride, padding, dilation,
                deconv_result
            )
        )

    save_oracle_result('deconv', deconv_oracle) 


conv_oracle_generator()
deconv_oracle_generator()