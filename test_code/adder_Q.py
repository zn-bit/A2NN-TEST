'''
量化
weight:scale>0 part median
input have << and >>
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
from torch.autograd.function import InplaceFunction, Function


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


def quantize_conv(x, num_bits, min_value=None, max_value=None, num_chunks=None, stochastic=False, inplace=False):
    qmax = 2. ** (num_bits - 1) - 1.
    scale = max(abs(max_value), abs(min_value)) / qmax
    scale = max(scale, 1e-8)
    scale_index = np.log2(scale).round()
    scale = 2. ** scale_index
    return UniformQuantize_conv().apply(x, num_bits, scale_index, scale, stochastic, inplace,
                                        num_chunks), scale, scale_index


def quantize_conv_new(x, num_bits, scale_index, scale, num_chunks=None, stochastic=False, inplace=False):
    return UniformQuantize_conv().apply(x, num_bits, scale_index, scale, stochastic, inplace,
                                        num_chunks)


class UniformQuantize_conv(InplaceFunction):
    @classmethod
    def forward(cls, ctx, input, num_bits, scale_index=None, scale=None, stochastic=False,
                inplace=False, num_chunks=None, out_half=False):
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.stochastic = stochastic
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        qmin = -2. ** (num_bits - 1) + 1.
        qmax = 2. ** (num_bits - 1) - 1.
        ctx.scale = scale
        if scale_index < 0:
            output = output << abs(scale_index)
        ctx.scale_index = scale_index
        output.clamp_(qmin, qmax).round_()  # quantize
        return output
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output / ctx.scale
        return grad_input, None, None, None, None, None, None, None


def quantize_bias(x, num_bits, scale_i=None, scale_i_index=None):
    return UniformQuantize_bias().apply(x, num_bits, scale_i, scale_i_index)


class UniformQuantize_bias(InplaceFunction):
    @classmethod
    def forward(cls, ctx, input, num_bits, scale_i=None, scale_i_index=None):
        ctx.num_bits = num_bits
        output = input.clone()
        qmin = -2. ** (num_bits - 1) + 1.
        qmax = 2. ** (num_bits - 1) - 1.
        ctx.scale_i = scale_i
        ctx.scale_i_index = scale_i_index
        # output = output / scale_i
        output = output << abs(scale_i_index)
        output = output.clamp_(qmin, qmax).round_()  # quantize
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_input = grad_output
        grad_input = grad_output / ctx.scale_i
        return grad_input, None, None, None, None, None, None, None, None


class QuantMeasure_conv(nn.Module):  # 对特征图mini_batch统计量化
    def __init__(self, num_bits, momentum=0.9):
        super(QuantMeasure_conv, self).__init__()
        self.register_buffer('running_min', -1.0 * torch.ones(1))
        self.register_buffer('running_max', 1.0 * torch.ones(1))
        self.momentum = momentum
        self.num_bits = num_bits

    def forward(self, input):
        if self.training:
            min_value = input.detach().view(
                input.size(0), -1).min(-1)[0].mean()
            max_value = input.detach().view(
                input.size(0), -1).max(-1)[0].mean()
            self.running_min.mul_(torch.FloatTensor([self.momentum]).cuda()).add_(
                min_value.data * (torch.FloatTensor([1 - self.momentum]).cuda()))
            self.running_max.mul_(torch.FloatTensor([self.momentum]).cuda()).add_(
                max_value.data * torch.FloatTensor([1 - self.momentum]).cuda())
        else:
            min_value = self.running_min
            max_value = self.running_max
        return quantize_conv_input(input, self.num_bits, min_value=float(min_value), max_value=float(max_value),
                             num_chunks=16)


def quantize_conv_input(x, num_bits, min_value=None, max_value=None, num_chunks=None, stochastic=False, inplace=False):
    qmax = 2. ** (num_bits - 1) - 1.
    scale = max(abs(max_value), abs(min_value)) / qmax
    scale = max(scale, 1e-8)
    scale_index = np.log2(scale).round()
    scale = 2 ** scale_index
    return UniformQuantize_conv_input().apply(x, num_bits, scale_index, scale, stochastic, inplace,
                                        num_chunks), scale, scale_index


class UniformQuantize_conv_input(InplaceFunction):
    @classmethod
    def forward(cls, ctx, input, num_bits, scale_index=None, scale=None, stochastic=False,
                inplace=False, num_chunks=None, out_half=False):
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.stochastic = stochastic
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        qmin = -2. ** (num_bits - 1) + 1.
        qmax = 2. ** (num_bits - 1) - 1.
        ctx.scale = scale
        if scale_index < 0:
            output = output << abs(scale_index)
        else:
            output = output >> scale_index
        ctx.scale_index = scale_index
        output.clamp_(qmin, qmax).round_()  # quantize
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output / ctx.scale
        return grad_input, None, None, None, None, None, None, None


def dequantize_conv(x, num_bits, scale_i=None, scale_i_index=None, num_chunks=None, stochastic=False, inplace=False):
    return deUniformQuantize_conv().apply(x, num_bits, scale_i, scale_i_index, stochastic, inplace, num_chunks)


class deUniformQuantize_conv(InplaceFunction):
    @classmethod
    def forward(cls, ctx, input, num_bits, scale_i=None, scale_i_index=None,
                stochastic=False, inplace=False, num_chunks=None):
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.stochastic = stochastic
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        output = output.round_().clamp_(-2 ** 31 + 1, 2 ** 31 - 1)  # quantize
        ctx.scale_i = scale_i
        ctx.scale_i_index = scale_i_index
        output = output >> abs(scale_i_index)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output * ctx.scale_i
        return grad_input, None, None, None, None, None, None, None, None


class deQuantMeasure_conv(nn.Module):
    """docstring for QuantMeasure."""
    def __init__(self, num_bits):
        super(deQuantMeasure_conv, self).__init__()
        self.num_bits = num_bits

    def forward(self, input, scale_i, scale_i_index):
        return dequantize_conv(input, self.num_bits, scale_i, scale_i_index)


class Qadder2d(nn.Module):
    """docstring for Qadder2d."""
    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=0, bias=False, num_bits=8,
                 momentum=0.9):
        super(Qadder2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.weight = torch.nn.Parameter(
            nn.init.normal_(torch.randn(output_channel, input_channel, kernel_size, kernel_size)))
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(nn.init.uniform_(torch.zeros(output_channel)))
        self.num_bits = num_bits
        self.quantize_input = QuantMeasure_conv(self.num_bits, momentum=momentum)
        self.dequantize_input = deQuantMeasure_conv(self.num_bits)
        self.quantize_median = 2. ** (self.num_bits - 2) - 1.

    def forward(self, x):
        qinput, scale_input, scale_index_input = self.quantize_input(x)
        qadder, scale_adder, scale_index_adder = quantize_conv(self.weight, num_bits=self.num_bits,
                                                               min_value=float(self.weight.min()),
                                                               max_value=float(self.weight.max()))
        ###############################
        if scale_index_adder > 0:
            p_idx = self.weight > 0
            n_idx = self.weight < 0
            adder_p_median = float(torch.median(torch.flatten(self.weight[p_idx])))
            adder_n_median = float(torch.median(torch.flatten(self.weight[n_idx])))
            scale_adder = max(abs(adder_p_median), abs(adder_n_median)) / self.quantize_median
            scale_adder = max(scale_adder, 1e-8)
            scale_index_adder = np.log2(scale_adder).round()
            scale_adder = 2. ** scale_index_adder
            qadder = quantize_conv_new(self.weight, num_bits=self.num_bits, scale_index=scale_index_adder,
                                       scale=scale_adder)
        else:
            scale_index_adder = scale_index_adder
            scale_adder = scale_adder
            qadder = qadder
        ###############################
        if scale_index_input < scale_index_adder:
            scale_i = scale_input
            scale_i_index = scale_index_input
            qadder = qadder << int(scale_index_adder - scale_index_input)
        elif scale_index_input == scale_index_adder:
            scale_i = scale_input
            scale_i_index = scale_index_input
        else:
            scale_i = scale_adder
            scale_i_index = scale_index_adder
            qinput = qinput << int(scale_index_input - scale_index_adder)
        qoutput = Adder2DFunction.apply(qinput,
                                        qadder,
                                        self.kernel_size,
                                        self.stride, self.padding)
        if self.bias:
            qbias = quantize_bias(self.b, num_bits=32, scale_i=scale_i, scale_i_index=scale_i_index)
            qoutput += qbias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        output = self.dequantize_input(qoutput, scale_i, scale_i_index)
        return output