'''
Refer to AdderNet code.
Efficient CUDA implementation for AdderNet training.
'''
import torch
import torch.nn as nn
# import adder_cuda
import numpy as np
from torch.autograd import Function
# from adder.quantize import quantize, quantize_grad, QuantMeasure, calculate_qparams
from torch.nn.modules.utils import _pair
from torch.utils.cpp_extension import load
import adder_cuda


# adder_cuda = load(
#   'adder_cuda', ['adder/adder_cuda.cpp', 'adder/adder_cuda_kernel.cu'], verbose=True)


def get_conv2d_output_shape(input, weight, stride, padding):
    n_filters, d_filter, h_filter, w_filter = weight.size()
    n_x, d_x, h_x, w_x = input.size()

    stride_h = int(stride)
    stride_w = int(stride)
    padding_h = int(padding)
    padding_w = int(padding)

    h_out = (h_x - h_filter + 2 * padding_h) // stride_h + 1
    w_out = (w_x - w_filter + 2 * padding_w) // stride_w + 1

    return (n_x, n_filters, h_out, w_out)


class Adder2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, kernel_size, stride, padding):
        ctx.save_for_backward(input, weight)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding

        kernel_size_h = kernel_size
        kernel_size_w = kernel_size
        stride_h = stride
        stride_w = stride
        padding_h = padding
        padding_w = padding

        output = input.new_zeros(
            get_conv2d_output_shape(input, weight, stride, padding))

        adder_cuda.forward(input,
                           weight,
                           output,
                           kernel_size_w, kernel_size_h,
                           stride_h, stride_w,
                           padding_h, padding_w)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None
        kernel_size, stride, padding = (
            ctx.kernel_size, ctx.stride, ctx.padding
        )

        kernel_size_h = kernel_size
        kernel_size_w = kernel_size
        stride_h = stride
        stride_w = stride
        padding_h = padding
        padding_w = padding

        # input
        if ctx.needs_input_grad[0]:
            grad_input = torch.zeros_like(input)
            adder_cuda.backward_input(grad_output,
                                      input,
                                      weight,
                                      grad_input,
                                      kernel_size_w, kernel_size_h,
                                      stride_w, stride_h,
                                      padding_h, padding_w)

        # weight
        if ctx.needs_input_grad[1]:
            grad_weight = torch.zeros_like(weight)
            adder_cuda.backward_weight(grad_output,
                                       input,
                                       weight,
                                       grad_weight,
                                       kernel_size_w, kernel_size_h,
                                       stride_w, stride_h,
                                       padding_h, padding_w)
            grad_weight = 0.2 * np.sqrt(grad_weight.numel()) / torch.norm(grad_weight) * grad_weight
        return grad_input, grad_weight, None, None, None


class adder2d(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=0, bias=False):
        super(adder2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.weight = torch.nn.Parameter(
            nn.init.normal_(torch.randn(
                output_channel, input_channel, int(kernel_size), int(kernel_size))))
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(
                nn.init.uniform_(torch.zeros(output_channel)))
    def forward(self, input):
        output = Adder2DFunction.apply(input,
                                       self.weight,
                                       self.kernel_size,
                                       self.stride,
                                       self.padding)
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return output
