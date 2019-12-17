---
title: '[Canny][Laplacian][Sobel][Scharr]边缘检测'
categories:
  - [算法, 图像处理, 边缘检测]
  - [编程, 编程语言]
  - [编程, 代码库]
tags:
  - canny
  - laplacian
  - sobel
  - scharr
  - c++
  - opencv
abbrlink: e42851a1
date: 2019-11-22 16:57:11
---

边缘检测是图像处理的基本操作之一，其目的是去除图像多余信息，保留图像轮廓数据，以便后续的处理（检测、识别等等）

## 降噪

边缘检测常常受噪声影响，所以通常先进行平滑处理，常用高斯滤波进行操作。参考：[高斯滤波](https://www.zhujian.tech/posts/80b530f2.html)

## 求导

参考：

[图像梯度](https://blog.csdn.net/u012005313/article/details/84068249)

[[Sobel]图像求导](https://zj-image-processing.readthedocs.io/zh_CN/latest/opencv/[Sobel]%E5%9B%BE%E5%83%8F%E6%B1%82%E5%AF%BC.html)

[[Scharr]图像求导](https://zj-image-processing.readthedocs.io/zh_CN/latest/opencv/[Scharr]%E5%9B%BE%E5%83%8F%E6%B1%82%E5%AF%BC.html)

[[Laplacian]图像求导](https://zj-image-processing.readthedocs.io/zh_CN/latest/opencv/[Laplacian]%E5%9B%BE%E5%83%8F%E6%B1%82%E5%AF%BC.html)

由于轮廓出现在像素值剧烈变化的位置，所以通过求导方式可以有效的保留轮廓信息。`OpenCV`实现了多个近似求导的算子，常用的一阶求导方法有`Sobel/Scharr`，以及二阶求导方法`Laplacian`

* 对于`Sobel`和`Scharr`而言，其均组合了平滑和差分功能，只不过相比较而言`Scharr`模板的中间系数较高，所以计算结果更加近似梯度计算
* 对于一阶求导和二阶求导算子而言，二阶导数除了能够找到候选的边缘像素，而且更容易区分像素变化的方向（递增和递减）以及剧烈程度。`OpenCV`提供的`Laplacian`算子模板仅专注于差分功能，能够得到更精细的边缘效果

## 边缘检测

相对于`Sobel/Scharr/Laplacian`算子，`Canny`算子是一个多步骤（包含滤波/求导/阈值等）组合在一起的边缘检测算法，其边缘检测实现效果也更加好

参考：[[Canny]边缘检测](https://zj-image-processing.readthedocs.io/zh_CN/latest/opencv/[Canny]%E8%BE%B9%E7%BC%98%E6%A3%80%E6%B5%8B.html)

## 示例

对同一张图像进行高斯滤波（$5\times 5, \sigma=1.4$）和灰度转换后，分别进行`Sobel/Scharr/Laplacian/Canny`检测

```
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat image, src, src_gray, grad;
int ksize = 3;
double scale = 1;
double delta = 0;
int ddepth = CV_16S;
int lowThreshold = 40;
int highThreshold = 120;

void onSobel(int, void *) {
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Sobel(src_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT); // x方向求导
    Sobel(src_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT); // y方向求导

    // converting back to CV_8U
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);                     // 近似计算图像梯度

    const string winname = "Sobel Edge Detector";
    imshow(winname, grad);
}

void onScharr(int, void *) {
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Scharr(src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);       // x方向求导
    Scharr(src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);       // y方向求导

    // converting back to CV_8U
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);                     // 近似计算图像梯度

    const string winname = "Scharr Edge Detector";
    imshow(winname, grad);
}

void onLaplacian(int, void *) {
    Mat grad, abs_grad;
    Laplacian(src_gray, grad, ddepth, ksize, scale, delta, BORDER_DEFAULT);

    // converting back to CV_8U
    convertScaleAbs(grad, abs_grad);

    const string winname = "Laplacian Edge Detector";
    imshow(winname, abs_grad);
}

void onCanny(int, void *) {
    Mat dst, detected_edges;

    Canny(src, detected_edges, lowThreshold, highThreshold, ksize);
    dst = Scalar::all(0);
    src.copyTo(dst, detected_edges);

    const string winname = "Canny Edge Detector";
    imshow(winname, dst);
}

int main(int argc, char **argv) {
    string imageName = "../lena.jpg";
    // As usual we load our source image (src)
    image = imread(imageName, IMREAD_COLOR); // Load an image
    // Check if image is loaded fine
    if (image.empty()) {
        printf("Error opening image: %s\n", imageName.c_str());
        return 1;
    }

    // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    GaussianBlur(image, src, Size(5, 5), 1.4, 1.4, BORDER_DEFAULT);
    // Convert the image to grayscale
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    double t0 = cv::getTickCount();
    onSobel(0, nullptr);
    double t1 = cv::getTickCount();
    onScharr(0, nullptr);
    double t2 = cv::getTickCount();
    onLaplacian(0, nullptr);
    double t3 = cv::getTickCount();
    onCanny(0, nullptr);
    double t4 = cv::getTickCount();

    double tickFrequency = cv::getTickFrequency();

    cout << "sobel: " << (t1 - t0) / tickFrequency << endl;
    cout << "scharr: " << (t2 - t1) / tickFrequency << endl;
    cout << "laplacian: " << (t3 - t2) / tickFrequency << endl;
    cout << "canny: " << (t4 - t3) / tickFrequency << endl;

    waitKey(0);
    return 0;
}
```

设置模板大小为$3\times 3$，`scale=1, delta=0`，利用`L1`范数计算梯度

处理$512\times 512$大小图像`lena.jpg`如下：

```
sobel: 0.0693918
scharr: 0.0310751
laplacian: 0.0205793
canny: 0.0435819
```

![](/imgs/edge-detect/sobel-scharr.png)

![](/imgs/edge-detect/laplacian-canny.png)

处理$512\times 512$大小图像`baboon.jpg`如下：

```
sobel: 0.0643578
scharr: 0.0307268
laplacian: 0.0180599
canny: 0.0254213
```

![](/imgs/edge-detect/sobel-scharr-2.png)

![](/imgs/edge-detect/laplacian-canny-2.png)

## 分析

从试验结果中发现

1. `Sobel`算子的平滑效果优于`Scharr`算子，但是其求导精度低于后者
2. `Laplacian`算子能够比`Sobel/Scharr`算子得到更精细的边缘轮廓
3. 相比于`Laplacian`算子，`Canny`算子能够得到更加明确的边缘轮廓
4. `Laplacian`算子仅注重于差分操作，所以其计算时间小于`Sobel/Scharr`算子
5. `OpenCV`中的`Canny`算子使用多线程进行计算，所以其计算时间小于`Sobel/Scharr`算子
