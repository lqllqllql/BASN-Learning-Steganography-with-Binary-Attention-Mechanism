'''
定义的函数
Utilitiy Functions
'''

import PIL.Image
#  collections模块：提供不同类型的container；container是一个对象，用于存储不同对象，并提供一种访问所包含对象并循环访问他们的方法。
# https://www.geeksforgeeks.org/python-collections-module/
import collections
import math
import numpy as np
# scikit-image is a collection of algorithms for image processing.
# https://scikit-image.org/docs/stable/api/skimage.feature.html
# 如展示图像边缘函数
import skimage
# .feature是可调用的计算特征库，利用封装好的函数计算图像特征和常见特征
import skimage.feature
import torch

# 图像物体边缘函数
class Edges(object):
  def __call__(self, sample):
    # 图像像素进行归一化
    image = np.array(sample) / 255.
    
    # https://www.cnblogs.com/wlzy/p/7966525.html
    # https://welts.xyz/2022/01/22/soft_thresholding/
    # 软阈值函数：T为阈值，对输入信号，函数将较小的输入置为0，较大的输入减小输出，进行降噪
    # 软阈值:解决优化问题，类似一种降噪函数
    # 在.feature模块的函数中使用到
    soft_min = 0.05
    soft_max = 0.95
    
    # .canny()：边缘检测；输出是一个2维数组
    # skimage.color.rgb2gray(image):将彩色图像转换为灰度图像
    # ---canny_edges_mask为一个已经检测好物体边缘的灰度图像
    canny_edges_mask = skimage.feature.canny(skimage.color.rgb2gray(image))
    # 执行滞后阈值：将高于高阈值的所有点标记为边。后以递归方式将高于低阈值并有8个点链接的任意点标记为边。
    # np.zeros_like()：返回与给定图像具有相同形状和类型的零数组
    # ---初始化为最低阈值矩阵
    canny_edges = np.zeros_like(canny_edges_mask) + soft_min
    # 通过索引将指定位置的值替换成soft_max
    canny_edges[canny_edges_mask] = soft_max
    
    # 对象：归一化后的图像像素矩阵，以便增加数据集数量
    # torch.tensor()：将张量构造成列表和序列
    # image.transpose():改变序列【反转或置换数组的轴，返回修改后的数组】
    # ---对于2维数组，转置数组给出矩阵的转置
    # ---对于3维数组，交换轴上的数据：z轴换到x轴，x轴换到y轴，y轴换到z轴
    image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
    
    # 对象：边缘检测后的图像像素
    # .expand_dims()：函数通过指定位置插入新的轴来扩展数组形状
    # ---多插一个轴变成三维数组，再展成序列或列表
    canny_edges = torch.tensor(np.expand_dims(canny_edges, 0), dtype=torch.float32)

    return image, canny_edges

# 学习率
def adjust_learning_rate(learning_rate, optimizer, epoch):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  lr = learning_rate * (0.1**(epoch // 30))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

# 准确率
# input：predictions预测图像
def accuracy(predictions, labels, top_k=(1, )):
  """Computes the top-k(s) for the specified values of k"""
  with torch.no_grad():
    max_k = max(top_k)
    batch_size = labels.size(0)

    # torch.topk(input， k， dim=None， maximum=True， sorted=True， *， out=None)
    # ---沿着给定维度返回张量的最大元素
    _, pred = predictions.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    def _get_accuracy_from_correct(k):
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      return correct_k.mul_(100.0 / batch_size)

    return [_get_accuracy_from_correct(k) for k in top_k]


# DataParallel分布式训练
# state_dict：字典
def remove_data_parallel_prefix(state_dict):
  """Remove DataParallel Prefix for Model Restoring"""
  new_state_dict = collections.OrderedDict()#定义一个字典子类，会记录键的插入顺序。
  # k:索引或顺序，v:值
  for k, v in state_dict.items():
    # .startswith()：检查字符串是否以module.开头
    if k.startswith('module.'):
      k = k[7:]
    new_state_dict[k] = v
  return new_state_dict


def format_topk_info(meta, prob, topk, classname, filename):
  """format topk information"""
  topk_info = []
  for prob_i, topk_i in zip(prob, topk):
    words = meta[topk_i][b'words'].decode()
    topk_info.append('%s[%04d](%4.2f%%)' % (words, topk_i, prob_i * 100))
  topk_info = ', '.join(topk_info)
  return '%s(%s): %s' % (filename, classname, topk_info)


def image_grid(images, padding=3, pad_value=0):
  '''
  Images:
      numpy.ndarray, NHWC type
  '''
  batch_size = images.shape[0]
  nrows, ncols, ncnls = images.shape[1:]

  impcol = math.ceil(math.sqrt(batch_size))
  improw = math.ceil(batch_size / impcol)

  ret_rows = improw * nrows + padding * (improw + 1)
  ret_cols = impcol * ncols + padding * (impcol + 1)

  ret = np.ones((ret_rows, ret_cols, ncnls)) * pad_value

  for ridx in range(improw):
    for cidx in range(impcol):
      idx = ridx * impcol + cidx
      if idx >= batch_size:
        break
      img = images[idx]
      rlb = (padding + nrows) * ridx + padding
      rub = (padding + nrows) * (ridx + 1)
      clb = (padding + ncols) * cidx + padding
      cub = (padding + ncols) * (cidx + 1)
      ret[rlb:rub, clb:cub, :] = img

  return ret


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def reset(self):
    """Reset current meter"""
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    """Update current meter with val (n is the batch size)"""
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class DeNormalize(object):
  """De-Normalize a tensor image with mean and standard deviation.
  Inverse operation to torchvision.transforms.Normalize
  """

  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __call__(self, tensor):
    """
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
        Tensor: Normalized Tensor image.
    """
    if not torch.is_tensor(tensor) and tensor.ndimension() == 3:
      raise TypeError('tensor is not a torch image.')

    tensor = tensor.clone().detach()

    # This is faster than using broadcasting, don't change without benchmarking
    for t, m, s in zip(tensor, self.mean, self.std):
      t.mul_(s).add_(m)

    return tensor

  def __repr__(self):
    return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
