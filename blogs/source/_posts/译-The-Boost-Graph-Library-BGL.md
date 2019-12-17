---
title: '[译]The Boost Graph Library(BGL)'
categories:
  - - 翻译
  - - 数据结构
  - [编程, 编程语言]
  - [编程, 代码库]
tags:
  - c++
  - 图
  - stl
abbrlink: af977180
date: 2019-10-25 14:57:23
---

原文地址：[The Boost Graph Library (BGL)](https://www.boost.org/doc/libs/1_71_0/libs/graph/doc/index.html)

>Graphs are mathematical abstractions that are useful for solving many types of problems in computer science. Consequently, these abstractions must also be represented in computer programs. A standardized generic interface for traversing graphs is of utmost importance to encourage reuse of graph algorithms and data structures. Part of the Boost Graph Library is a generic interface that allows access to a graph's structure, but hides the details of the implementation. This is an “open” interface in the sense that any graph library that implements this interface will be interoperable with the BGL generic algorithms and with other algorithms that also use this interface. The BGL provides some general purpose graph classes that conform to this interface, but they are not meant to be the “only” graph classes; there certainly will be other graph classes that are better for certain situations. We believe that the main contribution of the The BGL is the formulation of this interface.

图是数学抽象，对解决计算机科学中的许多问题都很有用。因此，这些抽象也必须在计算机程序中表示出来。一个用于遍历图的标准化通用接口对于鼓励图算法和数据结构的重用至关重要。Boost图库的一部分是一个通用接口，它允许访问图的结构，但隐藏了实现的细节。这是一个"开放"接口，因为实现此接口的任何图库都可以与BGL通用算法和使用此接口的其他算法进行互操作。BGL提供了一些符合这个接口的通用图类，但它们并不是"唯一"的图类；当然还有其他图类更适合某些情况。我们相信BGL的主要贡献是这个接口的模式

>The BGL graph interface and graph components are generic, in the same sense as the Standard Template Library (STL) [2]. In the following sections, we review the role that generic programming plays in the STL and compare that to how we applied generic programming in the context of graphs.

BGL的图接口和图组件是通用的，其含义与标准模板库（STL）相同。在下面的小节中，我们将回顾泛型编程在STL中所起的作用，并将其与我们如何在图上下文中应用泛型编程进行比较

>Of course, if you are already familiar with generic programming, please dive right in! Here's the Table of Contents. For distributed-memory parallelism, you can also look at the Parallel BGL.

当然，如果你已经熟悉通用编程，请直接跳过去！这是[目录](https://www.boost.org/doc/libs/1_71_0/libs/graph/doc/table_of_contents.html)。对于分布式内存并行，您还可以查看[并行BGL](https://www.boost.org/doc/libs/1_71_0/libs/graph_parallel/doc/html/index.html)

>The source for the BGL is available as part of the Boost distribution, which you can download from here.

BGL的源代码是boost发行版的一部分，可以从这里[下载](http://sourceforge.net/project/showfiles.php?group_id=7586)

## How to Build the BGL

如何构建BGL

>DON'T! The Boost Graph Library is a header-only library and does not need to be built to be used. The only exceptions are the GraphViz input parser and the GraphML parser.

不需要！Boost图库是一个只包含头文件的库，不需要构建就可以使用。唯一的例外是[GraphViz输入解析器](https://www.boost.org/doc/libs/1_71_0/libs/graph/doc/read_graphviz.html)和[GraphML解析器](https://www.boost.org/doc/libs/1_71_0/libs/graph/doc/read_graphml.html)

>When compiling programs that use the BGL, be sure to compile with optimization. For instance, select “Release” mode with Microsoft Visual C++ or supply the flag -O2 or -O3 to GCC.

在编译使用BGL的程序时，一定要进行优化编译。例如，如果使用微软Visual C++，选择"Release"模式；如果使用GCC，提供-O2或-O3标志

## Genericity in STL

STL中的泛型

>There are three ways in which the STL is generic.

STL有三种通用方法

### Algorithm/Data-Structure Interoperability

算法/数据结构互操作性

>First, each algorithm is written in a data-structure neutral way, allowing a single template function to operate on many different classes of containers. The concept of an iterator is the key ingredient in this decoupling of algorithms and data-structures. The impact of this technique is a reduction in the STL's code size from O(M*N) to O(M+N), where M is the number of algorithms and N is the number of containers. Considering a situation of 20 algorithms and 5 data-structures, this would be the difference between writing 100 functions versus only 25 functions! And the differences continues to grow faster and faster as the number of algorithms and data-structures increase.

首先，每个算法都不倾向于具体数据结构，允许单个模板函数在许多不同的容器类上操作。迭代器的概念是算法和数据结构解耦的关键因素。这种技术的影响是将STL的代码大小从O(M*N)减少到O(M+N)，其中M是算法的数量，N是容器的数量。考虑到20个算法和5个数据结构的情况，这将是写100个函数和只写25个函数的区别！随着算法和数据结构数量的增加，这种差异继续以越来越快的速度增长。

### Extension through Function Objects

通过函数对象的扩展

>The second way that STL is generic is that its algorithms and containers are extensible. The user can adapt and customize the STL through the use of function objects. This flexibility is what makes STL such a great tool for solving real-world problems. Each programming problem brings its own set of entities and interactions that must be modeled. Function objects provide a mechanism for extending the STL to handle the specifics of each problem domain.

STL通用的第二种方式是它的算法和容器是可扩展的。用户可以通过使用函数对象来调整和定制STL。这种灵活性使得STL成为解决现实世界问题的伟大工具。每个编程问题都有自己的一组必须建模的实体和交互。函数对象提供了一种扩展STL以处理每个问题域的细节的机制

### Element Type Parameterization

元素类型参数化

>The third way that STL is generic is that its containers are parameterized on the element type. Though hugely important, this is perhaps the least “interesting” way in which STL is generic. Generic programming is often summarized by a brief description of parameterized lists such as std::list<T>. This hardly scratches the surface!

STL泛型的第三种方式是在元素类型上对其容器进行参数化。尽管这非常重要，但这可能是STL通用的最不"有趣"的方式。泛型编程通常通过对参数化列表（如`std::list<T>`）的简要描述来总结。这使得理解内部实现更加困难！ 

## Genericity in the Boost Graph Library

Boost图库中的泛型

>Like the STL, there are three ways in which the BGL is generic.

与STL一样，BGL有三种通用方式

### Algorithm/Data-Structure Interoperability

算法/数据结构互操作性

>First, the graph algorithms of the BGL are written to an interface that abstracts away the details of the particular graph data-structure. Like the STL, the BGL uses iterators to define the interface for data-structure traversal. There are three distinct graph traversal patterns: traversal of all vertices in the graph, through all of the edges, and along the adjacency structure of the graph (from a vertex to each of its neighbors). There are separate iterators for each pattern of traversal.

首先，BGL的图算法被写入一个接口，该接口抽象出特定图数据结构的细节。与STL一样，BGL使用迭代器定义数据结构遍历的接口。有三种不同的图遍历模式：遍历图中的所有顶点、遍历所有边以及沿着图的邻接结构（从一个顶点到它的每个邻接顶点）。每个遍历模式都有独立的迭代器

>This generic interface allows template functions such as breadth_first_search() to work on a large variety of graph data-structures, from graphs implemented with pointer-linked nodes to graphs encoded in arrays. This flexibility is especially important in the domain of graphs. Graph data-structures are often custom-made for a particular application. Traditionally, if programmers want to reuse an algorithm implementation they must convert/copy their graph data into the graph library's prescribed graph structure. This is the case with libraries such as LEDA, GTL, Stanford GraphBase; it is especially true of graph algorithms written in Fortran. This severely limits the reuse of their graph algorithms.

此通用接口允许模板函数（比如[breaddth_first_search](https://www.boost.org/doc/libs/1_71_0/libs/graph/doc/breadth_first_search.html)）处理各种图数据结构，从使用指针链接节点实现的图到在数组中编码的图。这种灵活性在图领域中尤其重要。图数据结构通常是为特定应用程序定制的。传统上，如果程序员想要重用一个算法实现，他们必须将他们的图数据转换/复制到图库的指定图结构中。LEDA、GTL、Stanford GraphBase等库就是这样；用Fortran编写的图算法尤其如此。这严重限制了它们的图算法的重用

>In contrast, custom-made (or even legacy) graph structures can be used as-is with the generic graph algorithms of the BGL, using external adaptation (see Section How to Convert Existing Graphs to the BGL). External adaptation wraps a new interface around a data-structure without copying and without placing the data inside adaptor objects. The BGL interface was carefully designed to make this adaptation easy. To demonstrate this, we have built interfacing code for using a variety of graph structures (LEDA graphs, Stanford GraphBase graphs, and even Fortran-style arrays) in BGL graph algorithms.

相比之下，定制的（甚至是遗留的）图结构可以与BGL的通用图算法一起使用，使用外部自适应（参见[如何将现有图转换为BGL](https://www.boost.org/doc/libs/1_71_0/libs/graph/doc/leda_conversion.html)）。外部自适应在数据结构周围包装一个新接口，而无需复制，也无需将数据放置在适配器对象中。BGL接口经过精心设计，使这种调整变得容易。为了演示这一点，我们构建了接口代码，用于在BGL图算法中使用各种图结构（LEDA图、Stanford GraphBase图，甚至Fortran样式的数组）

### Extension through Visitors

通过Visitors扩展

>Second, the graph algorithms of the BGL are extensible. The BGL introduces the notion of a visitor, which is just a function object with multiple methods. In graph algorithms, there are often several key “event points” at which it is useful to insert user-defined operations. The visitor object has a different method that is invoked at each event point. The particular event points and corresponding visitor methods depend on the particular algorithm. They often include methods like start_vertex(), discover_vertex(), examine_edge(), tree_edge(), and finish_vertex().

其次，BGL的图算法是可扩展的。BGL引入了visitor的概念，它只是一个带有多个方法的函数对象。在图算法中，通常有几个关键的"事件点"，在这些点上插入用户定义的操作是有用的。visitor对象有一个在每个事件点调用的不同方法。特定的事件点和相应的访问者方法取决于特定的算法。它们通常包括start_vertex()、discover_vertex()、examine_edge()、tree_edge()和finish_vertex()等方法

### Vertex and Edge Property Multi-Parameterization

顶点和边属性多参数化

>The third way that the BGL is generic is analogous to the parameterization of the element-type in STL containers, though again the story is a bit more complicated for graphs. We need to associate values (called “properties”) with both the vertices and the edges of the graph. In addition, it will often be necessary to associate multiple properties with each vertex and edge; this is what we mean by multi-parameterization. The STL std::list<T> class has a parameter T for its element type. Similarly, BGL graph classes have template parameters for vertex and edge “properties”. A property specifies the parameterized type of the property and also assigns an identifying tag to the property. This tag is used to distinguish between the multiple properties which an edge or vertex may have. A property value that is attached to a particular vertex or edge can be obtained via a property map. There is a separate property map for each property.

BGL提供的第三种通用的方式类似于STL容器中元素类型的参数化，不过对于图来说，这个过程还是有点复杂。我们需要将值（称为"属性"）与图的顶点和边相关联。此外，通常需要将多个属性与每个顶点和边关联起来；这就是我们所说的多参数化。STL std::list<T>类的元素类型有一个参数T。类似地，BGL图类具有顶点和边"属性"的模板参数。属性指定属性的参数化类型，并为属性指定标识标记。此标记用于区分边或顶点可能具有的多个属性。附加到特定顶点或边的特性值可以通过特性映射获得。每个属性都有一个单独的属性映射

>Traditional graph libraries and graph structures fall down when it comes to the parameterization of graph properties. This is one of the primary reasons that graph data-structures must be custom-built for applications. The parameterization of properties in the BGL graph classes makes them well suited for re-use.

传统的图库和图结构在对图的属性进行参数化时会出现故障。这是图数据结构必须为应用程序定制的主要原因之一。BGL图类中属性的参数化使它们非常适合重用

## Algorithms

算法

>The BGL algorithms consist of a core set of algorithm patterns (implemented as generic algorithms) and a larger set of graph algorithms. The core algorithm patterns are
    * Breadth First Search
    * Depth First Search
    * Uniform Cost Search

BGL算法由一组核心的算法模式（作为通用算法实现）和一组较大的图算法组成。核心算法模式是

* 广度优先搜索
* 深度优先搜索
* 成本一致搜索

>By themselves, the algorithm patterns do not compute any meaningful quantities over graphs; they are merely building blocks for constructing graph algorithms. The graph algorithms in the BGL currently include
    * Dijkstra's Shortest Paths
    * Bellman-Ford Shortest Paths
    * Johnson's All-Pairs Shortest Paths
    * Kruskal's Minimum Spanning Tree
    * Prim's Minimum Spanning Tree
    * Connected Components
    * Strongly Connected Components
    * Dynamic Connected Components (using Disjoint Sets)
    * Topological Sort
    * Transpose
    * Reverse Cuthill Mckee Ordering
    * Smallest Last Vertex Ordering
    * Sequential Vertex Coloring

算法模式本身并不计算图上任何有意义的量；它们只是构造图算法的构造块。BGL中的图算法目前包括

* Dijkstra最短路径
* Bellman-Ford最短路径
* Johnson全对最短路径
* Kruskal最小生成树
* Prim最小生成树
* 连通分量
* 强连通分量
* 动态连通分量 (使用不相交集)
* 拓扑排序
* 转置
* 反向卡特希尔麦基排序
* 最小最后顶点排序
* 序列顶点染色

## Data Structures

数据结构

>The BGL currently provides two graph classes and an edge list adaptor:
    * adjacency_list
    * adjacency_matrix
    * edge_list

BGL目前提供两个图类和一个边缘列表适配器：

* [邻接表](https://www.boost.org/doc/libs/1_71_0/libs/graph/doc/adjacency_list.html)
* [邻接矩阵](https://www.boost.org/doc/libs/1_71_0/libs/graph/doc/adjacency_matrix.html)
* [边列表](https://www.boost.org/doc/libs/1_71_0/libs/graph/doc/edge_list.html)

>The adjacency_list class is the general purpose “swiss army knife” of graph classes. It is highly parameterized so that it can be optimized for different situations: the graph is directed or undirected, allow or disallow parallel edges, efficient access to just the out-edges or also to the in-edges, fast vertex insertion and removal at the cost of extra space overhead, etc.

邻接表类是图类的通用"瑞士军刀"。它是高度参数化的，因此可以针对不同的情况进行优化：图是有向或无向的，允许或不允许平行边，有效地访问外部边或内部边，以额外的空间开销为代价快速插入和移除顶点等

>The adjacency_matrix class stores edges in a $|V| \times |V|$ matrix (where $|V|$ is the number of vertices). The elements of this matrix represent edges in the graph. Adjacency matrix representations are especially suitable for very dense graphs, i.e., those where the number of edges approaches $|V|^2$.

邻接矩阵类将边存储在$|V| \times |V|$矩阵中（其中$|V|$是顶点数）。这个矩阵的元素表示图中的边。邻接矩阵表示特别适合于非常稠密的图，即边数接近$|V|^2$的图

>The edge_list class is an adaptor that takes any kind of edge iterator and implements an Edge List Graph.

edge_list类是一个适配器，它接受任何类型的边迭代器并实现一个[边列表图](https://www.boost.org/doc/libs/1_71_0/libs/graph/doc/EdgeListGraph.html)