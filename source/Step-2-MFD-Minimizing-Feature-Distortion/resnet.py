'''
ResNet Model
'''

import torch
import torch.nn as nn

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# 基本残差块：输入和输出通道相同
class BasicBlock(nn.Module):
  expansion = 1
  
  # 默认：stride=1
  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    
    # 残差连接：downsample层的输入是x
    # downsample：通道数增加，尺寸减小
    if self.downsample is not None:
      residual = self.downsample(x)
    
    # 残差公式
    out += residual
    out = self.relu(out)

    return out

# 输入和输出通道不同
class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    # 1*1卷积
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    # 3*3卷积 stride=stride：2
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    # 1*1卷积
    self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64 # 输入卷积核个数64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    # 看_make_layer函数
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    # self.avgpool = nn.AvgPool2d(7, stride=1)
    # self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        # 初始化
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
  
  # 搭建网络层
  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    # 如果步幅不为1或者输入通道不等于输出通道*expansion：即进行下采样，通道数增加
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          nn.Conv2d(
              self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(planes * block.expansion),
      )

    layers = [] #网络层
    # 输入通道，输出通道，步幅，下采样
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    # x = self.layer3(x)
    # x = self.layer4(x)

    # x = self.avgpool(x)
    # x = x.view(x.size(0), -1)
    # x = self.fc(x)

    return x


def resnet18():
  """Constructs a ResNet-18 model."""
  return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34():
  """Constructs a ResNet-34 model."""
  return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet50():
  """Constructs a ResNet-50 model."""
  return ResNet(Bottleneck, [3, 4, 6, 3])


def resnet101():
  """Constructs a ResNet-101 model."""
  return ResNet(Bottleneck, [3, 4, 23, 3])


def resnet152():
  """Constructs a ResNet-152 model."""
  return ResNet(Bottleneck, [3, 8, 36, 3])
