'''
Variance Pooling 2D
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

# 定义一个类
class VariancePool2d(nn.Module):
  """ Variance Pool 2D module.

  Args:
    kernel_size: size of pooling kernel, int or 2-tuple
    stride: pool stride, int or 2-tuple
    padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
    same: override padding and enforce same padding, boolean
  """
  
  # 定义类中的方法
  def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
    super(VariancePool2d, self).__init__()
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.same = same
 
    # _k,_s每个维度的内核索引
    self._k = _pair(kernel_size)
    self._s = _pair(stride)
    self._p = _quadruple(padding)  # convert to l, r, t, b
    
    #计算随机变量的期望值
    # .AvgPool2d二维平均池化
    self.pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride)
  
  # 填充吗？
  # x是二维随机变量，特征图的像素
  # If padding is non-zero,then the input is implicitly(隐式) zero-padded on both sides
  # ---for padding number of points.
  # 输入图像大小(N,C,ih,iw)
  def _padding(self, x):
    if self.same:
      # ih,iw是x的高度和宽度
      # 下面是不同情况下，更新ih,iw
      ih, iw = x.size()[2:]
      if ih % self._s[0] == 0:
        ph = max(self._k[0] - self._s[0], 0)
      else:
        ph = max(self._k[0] - (ih % self._s[0]), 0)
      if iw % self._s[1] == 0:
        pw = max(self._k[1] - self._s[1], 0)
      else:
        pw = max(self._k[1] - (iw % self._s[1]), 0)
       
      pl = pw // 2
      pr = pw - pl
      pt = ph // 2
      pb = ph - pt
      padding = (pl, pr, pt, pb)
    else:
      padding = self._p
    return padding

  def forward(self, x):
    x = F.pad(x, self._padding(x), mode='reflect')
    # 方差池化公式
    x = self.pool(x * x) - self.pool(x)**2
    return x
