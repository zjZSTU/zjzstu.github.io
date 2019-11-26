---
title: '[OpenCV][纹理特征]如何计算不同方向下的高斯导数直方图'
categories:
  - - 图像处理
  - - 编程
tags:
  - opencv
  - c++
  - 直方图
  - 纹理特征
abbrlink: b03366b0
date: 2019-11-25 20:56:01
---

学习`SelectiveSearch`算法时候，其纹理特征需要计算类`SIFT`特征，实现方式是计算每张图片`8`个方向上`10 bin`大小的高斯导数直方图

>$S_{texture}(r_{i}, r_{j})$ measures texture similarity. We represent texture using fast SIFT-like measurements as SIFT itself works well for material recognition [20]. We take Gaussian derivatives in eight orientations using $σ = 1$ for each colour channel. For each orientation for each colour channel we extract a histogram using a bin size of 10. This leads to a texture histogram $T_{i} = {t_{i}^{1}, ..., t_{i}^{n}}$ for each region $r_{i}$ with dimensionality $n = 240$ when three colour channels are used. Texture histograms are normalised using the $L_{1}$ norm. Similarity is measured using histogram intersection:

`OpenCV`实现了`SelectiveSearch`算法，其通过图像旋转、`Scharr`滤波以及手动计算直方图的方式完成了纹理特征的计算。之前没有思考过如何完成不同方向下导数直方图的计算，学习里面代码实现不同方向下的导数直方图计算

源码地址：`opencv_contrib/modules/ximgproc/src/selectivesearchsegmentation.cpp`

## 完整流程

1. 计算高斯导数
2. 计算直方图
3. 直方图连接

## 计算高斯导数

需要分别计算`8`个方向上的高斯导数，分别是

1. `x`轴正/负方向
2. `y`轴正/负方向
3. 图像逆时针`45`度旋转后`x`轴正/负方向
4. 图像逆时针`45`度旋转后`y`轴正/负方向

使用函数`Scharr`对图像进行高斯求导，然后通过阈值函数`threshold`获取正/负方向的求导结果

完成上述操作后将结果进行标准化（`[0,255]`），以便后续直方图的计算。实现代码如下：

```
/**
 * 计算8个方向的高斯导数
 * @param src CV_8UC1
 * @param gauss_vector
 */
void calc_8_direction_guass(const Mat &src, vector<Mat> &gauss_vector) {
//    cout << src.channels() << endl;

    Mat gauss, gauss_pos, gauss_neg;
    Mat rotated, rotated_gauss;
    Mat rotated_gauss_tmp;

    // x轴，向左/向右
    Scharr(src, gauss, CV_32F, 1, 0);
    threshold(gauss, gauss_pos, 0, 0, THRESH_TOZERO);
    threshold(gauss, gauss_neg, 0, 0, THRESH_TOZERO_INV);
    gauss_vector.emplace_back(gauss_pos);
    gauss_vector.emplace_back(gauss_neg);

    // y轴，向上/向下
    gauss.release();
    Scharr(src, gauss, CV_32F, 0, 1);

    gauss_pos.release();
    gauss_neg.release();
    threshold(gauss, gauss_pos, 0, 0, THRESH_TOZERO);
    threshold(gauss, gauss_neg, 0, 0, THRESH_TOZERO_INV);
    gauss_vector.emplace_back(gauss_pos);
    gauss_vector.emplace_back(gauss_neg);

    // 逆时针旋转45度
    Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    Mat rot = cv::getRotationMatrix2D(center, 45.0, 1.0);
    Rect bbox = cv::RotatedRect(center, src.size(), 45.0).boundingRect();
    rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
    rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;
    warpAffine(src, rotated, rot, bbox.size());
//    cout << rotated.size() << endl;

    // 计算x轴方向导数
    Scharr(rotated, rotated_gauss, CV_32F, 1, 0);

    // 顺时针旋转45度，获取原先图像大小
    center = Point((int) (rotated.cols / 2.0), (int) (rotated.rows / 2.0));
    rot = cv::getRotationMatrix2D(center, -45.0, 1.0);
    warpAffine(rotated_gauss, rotated_gauss_tmp, rot, bbox.size());
    gauss = rotated_gauss_tmp(Rect((bbox.width - src.cols) / 2,
                                   (bbox.height - src.rows) / 2, src.cols, src.rows));
    gauss_pos.release();
    gauss_neg.release();
    threshold(gauss, gauss_pos, 0, 0, THRESH_TOZERO);
    threshold(gauss, gauss_neg, 0, 0, THRESH_TOZERO_INV);
    gauss_vector.emplace_back(gauss_pos);
    gauss_vector.emplace_back(gauss_neg);

    // 重复上一步骤
    rotated_gauss.release();
    Scharr(rotated, rotated_gauss, CV_32F, 0, 1);

    // 顺时针旋转45度，获取原先图像大小
    center = Point((int) (rotated.cols / 2.0), (int) (rotated.rows / 2.0));
    rot = cv::getRotationMatrix2D(center, -45.0, 1.0);
    warpAffine(rotated_gauss, rotated_gauss_tmp, rot, bbox.size());
    gauss = rotated_gauss_tmp(Rect((bbox.width - src.cols) / 2,
                                   (bbox.height - src.rows) / 2, src.cols, src.rows));
    gauss_pos.release();
    gauss_neg.release();
    threshold(gauss, gauss_pos, 0, 0, THRESH_TOZERO);
    threshold(gauss, gauss_neg, 0, 0, THRESH_TOZERO_INV);
    gauss_vector.emplace_back(gauss_pos);
    gauss_vector.emplace_back(gauss_neg);

    // Normalisze gaussiaans in 0-255 range (for faster computation of histograms)
    // 缩放图像到0-255，方便直方图计算
    for (int i = 0; i < 8; i++) {
        double hmin, hmax;
        minMaxLoc(gauss_vector[i], &hmin, &hmax);

        Mat tmp;
        gauss_vector[i].convertTo(tmp, CV_8U,
                                  255 / (hmax - hmin),
                                  -255 * hmin / (hmax - hmin));
        gauss_vector[i] = tmp;
    }
}
```

* 进行阈值函数操作后直接放入向量中，所以求导图像有正有负
* 标准化公式如下：

$$
y = \frac{255*(x-hmin)}{hmax-hmin}
$$

## 计算直方图

参考：[直方图](https://www.zhujian.tech/posts/f1eacfb6.html#more)

```
/**
 * 计算颜色直方图，图像取值固定为[0, 255]
 * @param src CV_8UC1或CV_8UC3大小图像
 * @param histograms 直方图向量
 * @param bins 直方图大小
 */
void calc_color_hist(const Mat &src, vector<Mat> &histograms, int bins) {
    int channels = src.channels();
    vector<Mat> img_planes;
    if (channels == 3) {
        split(src, img_planes);
    } else {
        // gray
        img_planes.emplace_back(src);
    }

    float range[] = {0, 256}; //the upper boundary is exclusive
    const float *histRange = {range};
    bool uniform = true, accumulate = false;

    for (int i = 0; i < channels; i++) {
        Mat hist, tranpose_hist;
        calcHist(&img_planes[i], 1, nullptr, Mat(), hist, 1, &bins, &histRange, uniform, accumulate);
        //转置图像，得到一行数据
        transpose(hist, tranpose_hist);
        histograms.emplace_back(tranpose_hist);
    }
}
```

调用`OpenCV`函数`calcHist`得到的是`N`列大小的矩阵，为方便后续计算，将其转置成单行矩阵

## 直方图连接

彩色图像有`3`个通道，每个通道有`8`个方向求导，共得到`3x8=24`个直方图

对每个求导图像计算`10 bin`大小的直方图，所以得到的纹理特征维数是`24x10=240`维

在连接各个直方图之前，可以先对其进行标准化（`[0,1]`），以便后续操作

```
int main() {
    Mat src = imread("../lena.jpg");
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    vector<Mat> img_planes;
    split(src, img_planes);

    // 得到3x8=24个高斯求导图像，取值范围在[0,255]
    vector<Mat> gauss_vectors;
    for (const Mat &img: img_planes) {
        vector<Mat> gauss_vector;
        calc_8_direction_guass(img, gauss_vector);
        gauss_vectors.insert(gauss_vectors.end(), gauss_vector.begin(), gauss_vector.end());
    }

    // 计算10 bin大小直方图
    vector<Mat> hists;
    for (const Mat &img: gauss_vectors) {
        vector<Mat> hist;
        calc_color_hist(img, hist, 10);
        hists.insert(hists.end(), hist.begin(), hist.end());
    }

    // 归一化直方图
    for (const Mat &img: hists) {
        Mat dst;
        img.convertTo(dst, CV_32F, 1.0 / sum(img)[0]);
        cout << dst << endl;
    }

    // 按行连接所有矩阵
    cv::Mat out;
    cv::vconcat(hists, out);
    cout << out << endl;

    return 0;
}
```