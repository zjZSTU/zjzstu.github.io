---
title: PPM文件解析
categories:
  - [数据, 数据格式]
  - [编程, 编程语言]
  - [编程, 代码库]
tags:
  - ppm
  - python
  - argparse
  - numpy
  - opencv
abbrlink: 6cbcc636
date: 2019-08-10 13:53:08
---

最近进行图像处理时遇到[PPM](http://netpbm.sourceforge.net/doc/ppm.html)文件，其格式与`PGM`文件类似，参考[Python pgm解析和格式转换](https://blog.csdn.net/u012005313/article/details/83685584)进行`PPM`文件格式解析以及图像格式转换

## PPM

`PPM`(`Portable Pixel Map`，便携式像素地图)文件是`Netpbm`开源工程设计的一种图像格式，除了`ppm`外，还有`pbm，pgm`。`PPM`文件由一个或多个`PPM`图像序列组成。在图像之前、之后或之间没有数据、分隔符或填充

每个`PPM`文件按序包含如下信息：

```
1. A "magic number" for identifying the file type. A ppm image's magic number is the two characters "P6".
2. Whitespace (blanks, TABs, CRs, LFs).
3. A width, formatted as ASCII characters in decimal.
4. Whitespace.
5. A height, again in ASCII decimal.
6. Whitespace.
7. The maximum color value (Maxval), again in ASCII decimal. Must be less than 65536 and more than zero.
8. A single whitespace character (usually a newline).
9. A raster of Height rows, in order from top to bottom. Each row consists of Width pixels, in order from left to right. Each pixel is a triplet of red, green, and blue samples, in that order. Each sample is represented in pure binary by either 1 or 2 bytes. If the Maxval is less than 256, it is 1 byte. Otherwise, it is 2 bytes. The most significant byte is first.

1. 用于识别文件类型的“幻数”。PPM图像的幻数是两个字符“P6”。
2. 空格（blanks, TABs, CRs, LFs）。
3. 宽度，格式为ASCII十进制数字。
4. 空格。
5. 高度，同样为ASCII十进制数字。
6. 空格。
7. 最大灰度值（Maxval），同时是ASCII十进制。范围为[0，6536]。
8. 单个空白字符（通常是换行符）。
9. 按从上到下的顺序排列的高度行光栅。每行由宽度大小像素组成，从左到右依次排列。每一个像素都是红、绿、蓝三色样本的三重组合。每个样本用纯二进制表示，用1或2个字节表示。如果maxval小于256，则为1字节。否则，它是2个字节。最重要的字节是第一个。
```

*注意 1：以`#`开头的字符串可能是注释，与[PBM](http://netpbm.sourceforge.net/doc/pbm.html)相同*

*注意 2：图像数据集是连续存放的*

使用`vim`打开`PPM`文件，初始文件信息如下：

```
P6
1440 1080
255
...
```

### Plain PPM

还存在另一种格式的`PPM`文件，其差别如下：

```
1. There is exactly one image in a file.
2. The magic number is P3 instead of P6.
3. Each sample in the raster is represented as an ASCII decimal number (of arbitrary size).
4. Each sample in the raster has white space before and after it. There must be at least one character of white space between any two samples, but there is no maximum. There is no particular separation of one pixel from another -- just the required separation between the blue sample of one pixel from the red sample of the next pixel.
5. No line should be longer than 70 characters.

1. 文件中仅有单个图像。
2. 幻数是P3。
3. 栅格中的每个像素表示为ASCII十进制数（任意大小）。
4. 光栅中的每个样本前后都有空白。任何两个样本之间必须至少有一个空白字符，没有最多空白限制。一个像素与另一个像素之间没有特殊的分离——只是一个像素的蓝色样本与下一个像素的红色样本之间需要的分离。
5. 没有一行应该长于70个字符。
```

样本图像如下：

```
P3
# feep.ppm
4 4
15
 0  0  0    0  0  0    0  0  0   15  0 15
 0  0  0    0 15  7    0  0  0    0  0  0
 0  0  0    0  0  0    0 15  7    0  0  0
15  0 15    0  0  0    0  0  0    0  0  0
```

*注意：每行末尾都有一个换行符*

### 文件命名

通常以`.PPM`或者`.ppm`文件结尾

### PPM vs. PGM

`PPM`文件是`3`通道彩色图像文件，`PGM`是灰度图像文件

## 格式转换

可以使用库`Image`进行`PPM`格式与`PNG/JPEG`等其他格式转换

```
from PIL import Image

img = Image.open('test.PPM')
img.save('test2.png')
img.show()
```

也可依据图像格式读取文件信息后在转换成图像，完整代码如下：

```
# -*- coding: utf-8 -*-

# @Time    : 19-8-10 下午2:21
# @Author  : zj

from __future__ import print_function

import cv2
import time
import os
import operator
import numpy as np
import argparse
from PIL import Image

__author__ = 'zj'

image_formats = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']


def is_ppm_file(in_path):
    if not os.path.isfile(in_path):
        return False
    if in_path is not str and not in_path.lower().endswith('.ppm'):
        return False
    return True


def convert_ppm_by_PIL(in_path, out_path):
    if not is_ppm_file(in_path):
        raise Exception("%s 不是一个PPM文件" % in_path)
    # 读取文件
    im = Image.open(in_path)
    im.save(out_path)


def convert_ppm_P6(in_path, out_path):
    """
    将ppm文件转换成其它图像格式
    读取二进制文件，先读取幻数，再读取宽和高，以及最大值
    :param in_path: 输入ppm文件路径
    :param out_path: 输出文件路径
    """
    if not is_ppm_file(in_path):
        raise Exception("%s 不是一个PPM文件" % in_path)
    with open(in_path, 'rb') as f:
        # 读取两个字节 - 幻数，并解码成字符串
        magic_number = f.readline().strip().decode('utf-8')
        if not operator.eq(magic_number, "P6"):
            raise Exception("该图像有误")
        # 读取高和宽
        width, height = f.readline().strip().decode('utf-8').split(' ')
        width = int(width)
        height = int(height)
        # 读取最大值
        maxval = f.readline().strip()
        byte_array = np.array(list(f.readline()))
        img = byte_array.reshape((height, width, 3)).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, img)
        print('%s save ok' % out_path)


def convert_ppm_P6_batch(in_dir, out_dir, res_format):
    """
    批量转换PPM文件
    :param in_dir: ppm文件夹路径
    :param out_dir: 输出文件夹路径
    :param res_format: 结果图像格式
    """
    if not os.path.isdir(in_dir):
        raise Exception('%s 不是路径' % in_dir)
    if not os.path.isdir(out_dir):
        raise Exception('%s 不是路径' % out_dir)
    if not res_format in image_formats:
        raise Exception('%s 暂不支持' % res_format)
    file_list = os.listdir(in_dir)
    for file_name in file_list:
        file_path = os.path.join(in_dir, file_name)
        # 若为ppm文件路径，那么将其进行格式转换
        if is_ppm_file(file_path):
            file_out_path = os.path.join(out_dir, os.path.splitext(file_name)[0] + '.' + res_format)
            convert_ppm_P6(file_path, file_out_path)
        # 若为目录，则新建结果文件目录，递归处理
        elif os.path.isdir(file_path):
            file_out_dir = os.path.join(out_dir, file_name)
            if not os.path.exists(file_out_dir):
                os.mkdir(file_out_dir)
            convert_ppm_P6_batch(file_path, file_out_dir, res_format)
        else:
            pass
    print('batch operation over')


if __name__ == '__main__':
    script_start_time = time.time()

    parser = argparse.ArgumentParser(description='Format Converter - PPM')

    ### Positional arguments

    ### Optional arguments

    parser.add_argument('-i', '--input', type=str, help='Path to the ppm file')
    parser.add_argument('-o', '--output', type=str, help='Path to the result file')
    parser.add_argument('--input_dir', type=str, help='Dir to the ppm files')
    parser.add_argument('--output_dir', type=str, help='Dir to the result files')
    parser.add_argument('-f', '--format', default='png', type=str, help='result image format')
    parser.add_argument('-b', '--batch', action="store_true", default=False, help='Batch processing')

    args = vars(parser.parse_args())
    # print(args)
    in_path = args['input']
    out_path = args['output']

    isbatch = args['batch']
    in_dir = args['input_dir']
    out_dir = args['output_dir']
    res_format = args['format']

    if in_path is not None and out_path is not None:
        # 转换单个ppm文件格式
        convert_ppm_P6(in_path, out_path)
        # convert_ppm_by_PIL(in_path, out_path)
    elif isbatch:
        # 批量转换
        convert_ppm_P6_batch(in_dir, out_dir, res_format)
    else:
        print('请输入相应参数')

    print('Script took %s seconds.' % (time.time() - script_start_time,))
```

转换单个`PPM`图像为`PNG`图像

```
$ python ppm_converter.py -i test.ppm -o test2.png
test2.png save ok
Script took 0.2900853157043457 seconds.
```

批量转换`PPM`图像

```
$ python ppm_converter.py --input_dir 输入路径 --output_dir 输出路径 -f PNG -b
```