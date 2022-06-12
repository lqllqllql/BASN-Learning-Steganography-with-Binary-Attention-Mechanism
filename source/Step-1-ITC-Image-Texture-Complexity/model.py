'''
Model Definitions
'''
# pylint: disable=E1102
from functools import reduce

import operator

import torch
import torch.nn as nn

import numpy as np

import ops

# 判断对象类型是否存在
def check_isinstance(object_, types):
  """Check object_ is an instance of any of types"""
  return reduce(operator.or_, [isinstance(object_, t) for t in types])

# 初始化模型参数
# 参数的初始化关系到网络能否训练出好的结果或者以多快的速度收敛；保持数值稳定性。
# ---正向传播：将每个隐藏单元的参数初始化为相等的值，在正向传播时每个隐藏单元将根据相同的输入计算出相同的值，并传递至输出层；
# ---反向传播：每个隐藏单元的参数梯度值相等。参数在使用基于梯度的优化算法迭代后值依然相等。
#    ---在此情况下，无论隐藏单元有多少哥，隐藏层本质上只有一个隐藏单元在发挥作用。
def initialize_module(model, no_init_types=None):
  """Initialize a pytorch Module"""
  for m in model.modules():
    # 是否存在卷积层
    if check_isinstance(m, [nn.Conv2d, nn.ConvTranspose2d]):
      # Convolutions
      # 初始化函数初始化卷积层网络
      # .init.kaiming_normal(m,a=0,mode,nolinearity):使用正态分布（_normal_），用值填充输入的m
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # 是否存在BN层
    elif isinstance(m, nn.BatchNorm2d):
      # Normalizations
      # 初始化BN层：权重和偏置值
      # .init.constant_(m,val):使用val填充m
      nn.init.constant_(m.weight, 1)
      nn.init.constant_(m.bias, 0)
    # 是否存在激活函数
    elif check_isinstance(m, [nn.ReLU, nn.ELU, nn.Sigmoid]):
      # Activations
      pass
    # 是否存在Sequential模块
    elif check_isinstance(m, [type(model), nn.Sequential]):
      # Torch Types
      pass
    elif no_init_types and check_isinstance(m, no_init_types):
      # Customs
      pass
    else:
      # raise触发异常，触发后之后的程序将不再执行
      # RuntimeError：一般运行时的错误
      raise RuntimeError('Uninitialized layer: %s\n%s' % (type(m), m))

# ITC注意力模型
class Attentioner(nn.Module):
  # 定义方法
  def __init__(self):
    super(Attentioner, self).__init__()

    self.model = nn.Sequential(
        # 二维卷积：.Conv2d(in_channels,out_channels,kernel_size,stride,padding,...)
        # 第3个参数是卷积核的大小
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        # .ELU（alpha=1.0,inplace=False)---一种激活函数
        # 按元素应用指数线性单位函数：通过指数线性单元（ELU）进行快速准确地深度网络学习。
        # ---alpha:ELU的参数值；
        # ---inplace:can optionally do the operation in-place.
        nn.ELU(inplace=True),
        #
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ELU(inplace=True),
        #
        nn.Conv2d(64, 32, 3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ELU(inplace=True),
        #
        nn.Conv2d(32, 3, 3, stride=1, padding=1),
        nn.Sigmoid(),
    )

    initialize_module(self)

  def forward(self, x):
    return self.model(x)


class ImageSmoother(nn.Module):
  def __init__(self, kernel_size=7):
    super(ImageSmoother, self).__init__()
    self.kernel_size = kernel_size

    self.model = nn.Sequential(
        ops.MedianPool2d(kernel_size=self.kernel_size, same=True),
        # .Hardtanh：一种非线性激活函数
        nn.Hardtanh(min_val=0, max_val=1),
    )

  def forward(self, x):
    return self.model(x)
