'''
下载ILSVRC2012数据集
ILSVRC2012 Dataset Loader
'''
# io模块提供python用于处理I/O类型的主要工具；
# 每个具体流有其相应功能：只读，只写或读写，允许任意随机访问或允许顺序访问；
import io

import numpy as np
from PIL import Image

# lmdb模块：内存映射数据库。包含一个数据文件和一个锁文件。
# lmdb文件可以同时由多个进程打开，在访问数据的代码里引用lmdb库，访问时给文件路径即可；
# lmdb使用内存映射的方式访问文件，使得文件内寻址的开销很小，使得指针运算能实现。
import lmdb
# msgpack模块是一种高效的二进制序列化格式。允许在多种语言之间交换数据（如JSON）。
import msgpack

import torch.utils.data


class ILSVRC2012(torch.utils.data.Dataset):
  def __init__(self, path, transform=None, target_transform=None):
    self.mdb_path = path
    
    # 打开数据库文件
    self.env = lmdb.open(self.mdb_path, readonly=True)
    self.txn = self.env.begin()
    self.entries = self.env.stat()['entries']

    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return self.entries

  def __getitem__(self, idx):
    image_rawd = self.txn.get('{:08d}'.format(idx).encode())
    image_info = msgpack.unpackb(image_rawd, encoding='utf-8')
    with Image.open(io.BytesIO(image_info['image'])) as im:
      image = im.convert('RGB')
    target = image_info['label'] - 1  # ILSVRC2012 ID is in range [1, 1000]

    if not self.transform is None:
      image = self.transform(image)

    if not self.target_transform is None:
      target = self.target_transform(target)

    return image, target
