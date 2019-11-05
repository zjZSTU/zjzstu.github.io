---
title: 基于图的图像分割-OpenCV源码
categories:
  - - 机器学习
  - - 图像处理
  - - 数据结构
  - - 编程
tags:
  - 图像分割
  - c++
  - 图
  - 并查集
  - 最小生成树
  - opencv
  - Kruskal
abbrlink: '18052054'
date: 2019-11-05 14:59:00
---

`OpenCV`在模块`opencv_contrib`中实现了基于图的图像分割算法，其实现和作者提供的工程源码略有差别

下面首先解析源码，然后通过示例验证分割效果

* 官网参考文档：[cv::ximgproc::segmentation::GraphSegmentation Class Reference](https://docs.opencv.org/4.0.1/dd/d19/classcv_1_1ximgproc_1_1segmentation_1_1GraphSegmentation.html)
* 头文件`segmentation.hpp - /path/to/include/opencv4/opencv2/ximgproc/segmentation.hpp`
* 源文件`graphsegmentation.cpp - /path/to/opencv_contrib/modules/ximgproc/src/graphsegmentation.cpp`
* 实现示例`graphsegmentation_demo.cpp - /path/to/opencv_contrib/modules/ximgproc/samples/graphsegmentation_demo.cpp`

`OpenCV`源码比较复杂，抽取相应实现到[GraphLib/cplusplus/samples/graphsegmentation](https://github.com/zjZSTU/GraphLib/tree/master/cplusplus/samples/graphsegmentation)

## 命令空间

算法位于命令空间`cv::ximgproc::segmentation`中

```
namespace cv {
    namespace ximgproc {
        namespace segmentation {
```

## 并查集

`OpenCV`实现了并查集操作，定义了并查集元素类`PointSetElement`以及并查集操作类`PointSet`

```
class PointSetElement {
    public:
        int p;
        int size;

        PointSetElement() { }

        PointSetElement(int p_) {
            p = p_;
            size = 1;
        }
};

// An object to manage set of points, who can be fusionned
class PointSet {
    public:
        PointSet(int nb_elements_);
        ~PointSet();

        int nb_elements;

        // Return the main point of the point's set
        int getBasePoint(int p);

        // Join two sets of points, based on their main point
        void joinPoints(int p_a, int p_b);

        // Return the set size of a set (based on the main point)
        int size(unsigned int p) { return mapping[p].size; }

    private:
        PointSetElement* mapping;

};
```

对于`PointSetElement`而言，定义了分量大小$size$以及当前像素点在最小生成树中的父指针$p$

对于`PointSet`而言，有两个成员和`3`个函数

* `nb_elements`：分量个数
* `mapping`：点集元素指针
* `getBasePoint(int p)`：得到元素所属分量的根节点坐标
* `joinPoints(int p_a, int p_b)`：合并两个分量
* `size(unsigned int p)`：返回元素`p`所在分量个数

```
PointSet::PointSet(int nb_elements_) {
    nb_elements = nb_elements_;

    mapping = new PointSetElement[nb_elements];

    for ( int i = 0; i < nb_elements; i++) {
        mapping[i] = PointSetElement(i);
    }
}

PointSet::~PointSet() {
    delete [] mapping;
}

int PointSet::getBasePoint( int p) {

        int base_p = p;

    while (base_p != mapping[base_p].p) {
        base_p = mapping[base_p].p;
    }

    // Save mapping for faster acces later
    mapping[p].p = base_p;

    return base_p;
}

void PointSet::joinPoints(int p_a, int p_b) {

    // Always target smaller set, to avoid redirection in getBasePoint
    if (mapping[p_a].size < mapping[p_b].size)
        swap(p_a, p_b);

    mapping[p_b].p = p_a;
    mapping[p_a].size += mapping[p_b].size;

    nb_elements--;
}
```

* 在构造函数中，通过输入的参数`nb_elements_`创建指针空间，初始化每个点集元素的父指针指向自身
* 函数`getBasePoint`查询根节点，使用了路径压缩进行优化
* 函数`joinPoints`合并两个分量，累加两个分量个数到根节点。与工程实现不同的是，这里比较`size`大小进行合并

## 边

定义类`Edge`保存边信息

```
class Edge {
    public:
        int from;
        int to;
        float weight;

        bool operator <(const Edge& e) const {
            return weight < e.weight;
        }
};
```

包含两个顶点坐标以及边权重，同时重写比较函数，可作用于边集排序

## 图分割算法

`OpenCV`定义了一个图分割算法声明类`GraphSegemntation`以及一个图分割算法实现类`GraphSegmentationImpl`

### 声明

图分割算法声明类`GraphSegmentation`位于`segmentation.hpp`

```
class CV_EXPORTS_W GraphSegmentation : public Algorithm {
    public:
        /** @brief Segment an image and store output in dst
            @param src The input image. Any number of channel (1 (Eg: Gray), 3 (Eg: RGB), 4 (Eg: RGB-D)) can be provided
            @param dst The output segmentation. It's a CV_32SC1 Mat with the same number of cols and rows as input image, with an unique, sequential, id for each pixel.
        */
        CV_WRAP virtual void processImage(InputArray src, OutputArray dst) = 0;

        CV_WRAP virtual void setSigma(double sigma) = 0;
        CV_WRAP virtual double getSigma() = 0;

        CV_WRAP virtual void setK(float k) = 0;
        CV_WRAP virtual float getK() = 0;

        CV_WRAP virtual void setMinSize(int min_size) = 0;
        CV_WRAP virtual int getMinSize() = 0;
};

/** @brief Creates a graph based segmentor
    @param sigma The sigma parameter, used to smooth image
    @param k The k parameter of the algorythm
    @param min_size The minimum size of segments
    */
CV_EXPORTS_W Ptr<GraphSegmentation> createGraphSegmentation(double sigma=0.5, float k=300, int min_size=100);
```

声明了对外提供的接口，同时提供了创建图分割类对象的辅助函数`createGraphSegmentation`

### 实现

图分割算法实现类`GraphSegmentationImpl`位于`segmentation.hpp`，其继承了接口类`GraphSegmentation`并实现了分割算法

```
class GraphSegmentationImpl : public GraphSegmentation {
    public:
        GraphSegmentationImpl() {
            sigma = 0.5;
            k = 300;
            min_size = 100;
            name_ = "GraphSegmentation";
        }

        ~GraphSegmentationImpl() CV_OVERRIDE {
        };

        virtual void processImage(InputArray src, OutputArray dst) CV_OVERRIDE;

        virtual void setSigma(double sigma_) CV_OVERRIDE { if (sigma_ <= 0) { sigma_ = 0.001; } sigma = sigma_; }
        virtual double getSigma() CV_OVERRIDE { return sigma; }

        virtual void setK(float k_) CV_OVERRIDE { k = k_; }
        virtual float getK() CV_OVERRIDE { return k; }

        virtual void setMinSize(int min_size_) CV_OVERRIDE { min_size = min_size_; }
        virtual int getMinSize() CV_OVERRIDE { return min_size; }

        virtual void write(FileStorage& fs) const CV_OVERRIDE {
            fs << "name" << name_
            << "sigma" << sigma
            << "k" << k
            << "min_size" << (int)min_size;
        }

        virtual void read(const FileNode& fn) CV_OVERRIDE {
            CV_Assert( (String)fn["name"] == name_ );

            sigma = (double)fn["sigma"];
            k = (float)fn["k"];
            min_size = (int)(int)fn["min_size"];
        }

    private:
        double sigma;
        float k;
        int min_size;
        String name_;

        // Pre-filter the image
        void filter(const Mat &img, Mat &img_filtered);

        // Build the graph between each pixels
        void buildGraph(Edge **edges, int &nb_edges, const Mat &img_filtered);

        // Segment the graph
        void segmentGraph(Edge * edges, const int &nb_edges, const Mat & img_filtered, PointSet **es);

        // Remove areas too small
        void filterSmallAreas(Edge *edges, const int &nb_edges, PointSet *es);

        // Map the segemented graph to a Mat with uniques, sequentials ids
        void finalMapping(PointSet *es, Mat &output);
};
```

`public`函数包括

* `processImage`：图像分割

`private`函数包括：

* `filter`：高斯滤波
* `buildgraph`：创建边集
* `segmentGraph`：`Kruskal`算法得到最小生成树
* `filterSmallAreas`：合并小分量
* `finalMapping`：创建输出图

另外`createGraphSegmentation`创建了分割类对象

#### createGraphSegmentation

创建类对象，赋值高斯滤波参数`sigma`，阈值函数参数`k`，最小分量大小`min_size`，最后返回对象指针

```
Ptr<GraphSegmentation> createGraphSegmentation(double sigma, float k, int min_size) {

    Ptr<GraphSegmentation> graphseg = makePtr<GraphSegmentationImpl>();

    graphseg->setSigma(sigma);
    graphseg->setK(k);
    graphseg->setMinSize(min_size);

    return graphseg;
}
```

#### filter

首先将输入图像转换成浮点类型，再调用高斯滤波函数`GaussianBlur`进行处理

```
void GraphSegmentationImpl::filter(const Mat &img, Mat &img_filtered) {

    Mat img_converted;

    // Switch to float
    img.convertTo(img_converted, CV_32F);

    // Apply gaussian filter
    GaussianBlur(img_converted, img_filtered, Size(0, 0), sigma, sigma);
}
```

输入卷积核大小为$Size(0,0)$，参考[getGaussianKernel](https://www.zhujian.tech/posts/80b530f2.html)，表示根据`sigma`值计算卷积核大小

#### buildgraph

从左到右，从上到下的遍历像素点，计算当前顶点和**上/下/左/右**顶点的边

```
for (int delta = -1; delta <= 1; delta += 2) {
    for (int delta_j = 0, delta_i = 1; delta_j <= 1; delta_j++ || delta_i--) {

        int i2 = i + delta * delta_i;
        int j2 = j + delta * delta_j;
```

`i2/j2`取值为

```
i2 = -1 j2 = 0
i2 = 0 j2 = -1
i2 = 1 j2 = 0
i2 = 0 j2 = 1
```

边权重通过计算相邻像素点之间的$L2$距离获得

```
for ( int channel = 0; channel < nb_channels; channel++) {
    tmp_total += pow(p[j * nb_channels + channel] - p2[j2 * nb_channels + channel], 2);
}
```

创建的边集会出现重复边的情况（*对无向图而言，虽然通过属性`from/to`明确了初始点和终止点*），不过在后续操作中都会使用到

#### segmentGraph

通过`Kruskal`算法实现分量的合并。首先进行边集排序，类`Edge`重写了比较函数，所以按权值升序排序

```
std::sort(edges, edges + nb_edges);
```

然后创建并查集类`PointSet`

```
*es = new PointSet(img_filtered.cols * img_filtered.rows);
```

并设置阈值函数，初始时每个分量个数为`1`，所以阈值大小为`k`

```
float* thresholds = new float[total_points];

for (int i = 0; i < total_points; i++)
    thresholds[i] = k;
```

遍历所有边，判断两个顶点是否位于同一分量。如果不是，判断是否满足边界条件。如果不是，合并两分量，更新阈值并设置边权重为`0`

```
for ( int i = 0; i < nb_edges; i++) {
    int p_a = (*es)->getBasePoint(edges[i].from);
    int p_b = (*es)->getBasePoint(edges[i].to);

    if (p_a != p_b) {
        if (edges[i].weight <= thresholds[p_a] && edges[i].weight <= thresholds[p_b]) {
            (*es)->joinPoints(p_a, p_b);
            p_a = (*es)->getBasePoint(p_a);
            thresholds[p_a] = edges[i].weight + k / (*es)->size(p_a);

            edges[i].weight = 0;
        }
    }
}
```

*由于边集存在重复边的情况，所以将已使用的边权值设置为`0`之后，还有另一条相同的无向边存在*

#### filterSmallAreas

再次遍历所有边，合并小分量

```
void GraphSegmentationImpl::filterSmallAreas(Edge *edges, const int &nb_edges, PointSet *es) {
    for ( int i = 0; i < nb_edges; i++) {
        if (edges[i].weight > 0) {
            int p_a = es->getBasePoint(edges[i].from);
            int p_b = es->getBasePoint(edges[i].to);

            if (p_a != p_b && (es->size(p_a) < min_size || es->size(p_b) < min_size)) {
                es->joinPoints(p_a, p_b);
            }
        }
    }
}
```

#### finalMapping

本函数作用于最后的不同分量颜色设置，输入参数为合并操作后的点集`PointSet *es`以及单通道图像`Mat &output`。同一分量的像素点赋值同一个值，像素值从`0`开始递增

## 示例

实现代码如下：

```
#include "opencv2/ximgproc/segmentation.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace cv::ximgproc::segmentation;

Scalar hsv_to_rgb(Scalar);

Scalar color_mapping(int);

static void help() {
    std::cout << std::endl <<
              "A program demonstrating the use and capabilities of a particular graph based image" << std::endl <<
              "segmentation algorithm described in P. Felzenszwalb, D. Huttenlocher," << std::endl <<
              "             \"Efficient Graph-Based Image Segmentation\"" << std::endl <<
              "International Journal of Computer Vision, Vol. 59, No. 2, September 2004" << std::endl << std::endl <<
              "Usage:" << std::endl <<
              "./graphsegmentation_demo input_image output_image [simga=0.5] [k=300] [min_size=100]" << std::endl;
}

Scalar hsv_to_rgb(Scalar c) {
    Mat in(1, 1, CV_32FC3);
    Mat out(1, 1, CV_32FC3);

    float *p = in.ptr<float>(0);
    p[0] = (float) c[0] * 360.0f;
    p[1] = (float) c[1];
    p[2] = (float) c[2];

    cvtColor(in, out, COLOR_HSV2RGB);

    Scalar t;
    Vec3f p2 = out.at<Vec3f>(0, 0);
    t[0] = (int) (p2[0] * 255);
    t[1] = (int) (p2[1] * 255);
    t[2] = (int) (p2[2] * 255);

    return t;
}

Scalar color_mapping(int segment_id) {
    double base = (double) (segment_id) * 0.618033988749895 + 0.24443434;

    return hsv_to_rgb(Scalar(fmod(base, 1.2), 0.95, 0.80));
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 6) {
        help();
        return -1;
    }

    Ptr<GraphSegmentation> gs = createGraphSegmentation();
    if (argc > 3)
        gs->setSigma(atof(argv[3]));
    if (argc > 4)
        gs->setK((float) atoi(argv[4]));
    if (argc > 5)
        gs->setMinSize(atoi(argv[5]));
    if (!gs) {
        std::cerr << "Failed to create GraphSegmentation Algorithm." << std::endl;
        return -2;
    }

    Mat input, output, output_image;
    input = imread(argv[1]);
    if (!input.data) {
        std::cerr << "Failed to load input image" << std::endl;
        return -3;
    }
    gs->processImage(input, output);

    double min, max;
    minMaxLoc(output, &min, &max);

    int nb_segs = (int) max + 1;
    std::cout << nb_segs << " segments" << std::endl;
    output_image = Mat::zeros(output.rows, output.cols, CV_8UC3);

    uint *p;
    uchar *p2;
    for (int i = 0; i < output.rows; i++) {
        p = output.ptr<uint>(i);
        p2 = output_image.ptr<uchar>(i);

        for (int j = 0; j < output.cols; j++) {
            Scalar color = color_mapping(p[j]);
            p2[j * 3] = (uchar) color[0];
            p2[j * 3 + 1] = (uchar) color[1];
            p2[j * 3 + 2] = (uchar) color[2];
        }
    }
    imwrite(argv[2], output_image);
    std::cout << "Image written to " << argv[2] << std::endl;

    return 0;
}
```

首先解析命令行参数，创建图像分割类对象并初始化参数

然后图像分割函数进行基于图的图像分割，输出单通道灰度图像

最后将创建`3`通道图像并赋值，同一分量的像素点设置相同的值。与论文提供的实现不同，为了使得分量间的颜色更加有区别，进行`HSV`颜色空间和`RGB`颜色空间的转换

![](/imgs/基于图的图像分割-OpenCV源码/beach.png)

`sigma=0.5, k=500, min_size=50`

![](/imgs/基于图的图像分割-OpenCV源码/beach_opencv.png)

`sigma=0.5, k=300, min_size=100`

![](/imgs/基于图的图像分割-OpenCV源码/beach_opencv2.png)