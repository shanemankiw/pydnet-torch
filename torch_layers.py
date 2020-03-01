
import torch.utils.data
from torch.nn import functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.conv import _ConvTransposeMixin, _ConvNd
from torch.nn.modules.utils import _single, _pair, _triple
import sys


class Convtranspose2d_same(_ConvTransposeMixin, _ConvNd):
    '''
    a implementation of 
    padding same conv_tranpose2d
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(Convtranspose2d_same, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias)

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        output = conv_tranpose2d_same_padding(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
        output = F.leaky_relu(output, 0.2)

'''
class _ConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

 

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

 

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
'''

class Conv2d_same_leaky(_ConvNd):
    '''
    from CSDN
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, relu=True):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.relu = relu
        super(Conv2d_same_leaky, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        '''
        this is a important spot of our implementation
        '''
        out = conv2d_same_padding(input, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups)
        
        if self.relu:
            out = F.leaky_relu(out, 0.2)

        return out


def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)
    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])
    '''
    print('padding_cols is {}'.format(padding_cols))
    print('padding_rows is {}'.format(padding_rows))
    '''

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)


def conv_tranpose2d_same_padding(input, weight, bias=None, stride=1, padding=1, output_padding=1,dilation=1, groups=1):
    '''
    this needs to be done!
    '''

    if dilation != 1:
        sys.exit('Fuck off!')

    output_padding = stride - 1
    kernel_size = weight.size(2)

    padding = (kernel_size - stride + output_padding)/2
 
    return F.conv_transpose2d(input, weight, bias, stride,
                  padding,
                  output_padding = output_padding,
                  dilation=dilation, groups=groups)
