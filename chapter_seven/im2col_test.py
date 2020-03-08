"""
Created by 陈辰柄 
datetime:2020/3/8 16:42
Describe: image to column 讲图像转换成列方法测试
"""

from common.util import im2col
import numpy as np

x1=np.random.rand(1,3,7,7)
coll=im2col(x1,5,5,stride=1,pad=0)

print(coll.shape)