---
title: '[译]欢迎回到C++(现代C++)'
abbrlink: be56146c
date: 2019-08-25 13:36:14
categories:
- [翻译]
- [编程, 编程语言]
tags:
- c++
---

之前写的C++代码一直都是C语言风格，而C++标准经过多次的迭代后，已经拥有了一套自己的风格和实现。微软的这篇文档很好的反映了现代C++的精髓，原文地址：[Welcome Back to C++ (Modern C++)](https://docs.microsoft.com/en-us/cpp/cpp/welcome-back-to-cpp-modern-cpp?view=vs-2019)

## 译文

>C++ is one of the most widely used programming languages in the world. Well-written C++ programs are fast and efficient. The language is more flexible than other languages because you can use it to create a wide range of apps—from fun and exciting games, to high-performance scientific software, to device drivers, embedded programs, and Windows client apps. For more than 20 years, C++ has been used to solve problems like these and many others. What you might not know is that an increasing number of C++ programmers have folded up the dowdy C-style programming of yesterday and have donned modern C++ instead.

C++是世界上应用最广泛的编程语言之一。写得好的C++程序是快速和高效的。该语言比其他语言更灵活，因为您可以使用它创建各种各样的应用程序，从有趣和刺激的游戏，到高性能的科学软件，再到设备驱动程序、嵌入式程序和Windows客户端应用程序。20多年来，C++已经被用来解决这些问题以及许多其他问题。你可能不知道的是，越来越多的C++程序员已经折叠了昨天过时的C风格编程，并使用了现代C++取代

>One of the original requirements for C++ was backward compatibility with the C language. Since then, C++ has evolved through several iterations—C with Classes, then the original C++ language specification, and then the many subsequent enhancements. Because of this heritage, C++ is often referred to as a multi-paradigm programming language. In C++, you can do purely procedural C-style programming that involves raw pointers, arrays, null-terminated character strings, custom data structures, and other features that may enable great performance but can also spawn bugs and complexity. Because C-style programming is fraught with perils like these, one of the founding goals for C++ was to make programs both type-safe and easier to write, extend, and maintain. Early on, C++ embraced programming paradigms such as object-oriented programming. Over the years, features have been added to the language, together with highly-tested standard libraries of data structures and algorithms. It's these additions that have made the modern C++ style possible.

对C++的最初要求之一是与C语言的向后兼容性。从那时起，C++已经经过几次迭代 -- 最开始的C++可以称为C与类，然后是最初的C++语言规范，以及后续许多增强。由于这个传统，C++经常被称为多范式编程语言。在C++中，您可以进行纯程序化的C风格编程，包括原始指针、数组、空终止字符串、自定义数据结构和其他可以使性能良好的特征，但也可能导致错误和复杂性。因为C风格的编程充满了这样的危险，C++的一个创始目标是使程序既安全又容易编写、扩展和维护。早期，C++采用了面向对象编程等编程范例。多年来，该语言增加了一些特性，以及经过高度测试的数据结构和算法标准库。正是这些添加物使现代C++风格成为可能

>Modern C++ emphasizes:
>* Stack-based scope instead of heap or static global scope.
>* Auto type inference instead of explicit type names.
>* Smart pointers instead of raw pointers.
>* `std::string` and `std::wstring` types (see [\<string\>](https://docs.microsoft.com/en-us/cpp/standard-library/string?view=vs-2019)) instead of raw `char[]` arrays.
>* [C++ Standard Library](https://docs.microsoft.com/en-us/cpp/standard-library/cpp-standard-library-header-files?view=vs-2019) containers like `vector`, `list`, and `map` instead of raw arrays or custom containers. See [\<vector\>](https://docs.microsoft.com/en-us/cpp/standard-library/vector?view=vs-2019), [\<list\>](https://docs.microsoft.com/en-us/cpp/standard-library/list?view=vs-2019), and [\<map\>](https://docs.microsoft.com/en-us/cpp/standard-library/map?view=vs-2019).
>* C++ Standard Library [algorithms](https://docs.microsoft.com/en-us/cpp/standard-library/algorithm?view=vs-2019) instead of manually coded ones.
>* Exceptions, to report and handle error conditions.
>* Lock-free inter-thread communication using C++ Standard Library `std::atomic<>` (see [\<atomic\>](https://docs.microsoft.com/en-us/cpp/standard-library/atomic?view=vs-2019)) instead of other inter-thread communication mechanisms.
>* Inline [lambda functions](https://docs.microsoft.com/en-us/cpp/cpp/lambda-expressions-in-cpp?view=vs-2019) instead of small functions implemented separately.
>* Range-based for loops to write more robust loops that work with arrays, C++ Standard Library containers, and Windows Runtime collections in the form `for ( for-range-declaration : expression )`. This is part of the Core Language support. For more information, see [Range-based for Statement (C++)](https://docs.microsoft.com/en-us/cpp/cpp/range-based-for-statement-cpp?view=vs-2019).

现代C++强调：

* 基于栈的作用域，而不是堆或静态全局作用域
* 自动类型推断而不是显式类型名
* 智能指针而不是原始指针
* `std::string`和`std::wstring`类型（参见`<string>`）而不是原始`char[]`数组
* C++标准库容器，如`vector`、`list`和`map`，而不是原始数组或自定义容器。请参见`<vector>`、`<list>`和`<map>`
* C++标准库算法，而不是手工编写的算法
* 通过异常报告和处理错误情况
* 使用C++标准库`std::atomic<>`进行无锁线程间通信（参见`<atomic>`）而不是其他线程间通信机制
* 内联lambda函数而不是单独实现的小函数
* 使用基于范围的循环编写更健壮的循环代码，这些循环使用数组、C++标准库容器和Windows运行时集合`for ( for-range-declaration : expression )`。这是核心语言支持的一部分。有关更多信息，请参见基于范围的语句（C++）

>The C++ language itself has also evolved. Compare the following code snippets. This one shows how things used to be in C++:

C++语言本身也有了发展。比较以下代码段。这个例子显示了C++中的事物是如何使用的：

```
#include <vector>

void f()
{
    // Assume circle and shape are user-defined types
    circle* p = new circle( 42 );
    vector<shape*> v = load_shapes();

    for( vector<circle*>::iterator i = v.begin(); i != v.end(); ++i ) {
        if( *i && **i == *p )
            cout << **i << " is a match\n";
    }

    // CAUTION: If v's pointers own the objects, then you
    // must delete them all before v goes out of scope.
    // If v's pointers do not own the objects, and you delete
    // them here, any code that tries to dereference copies
    // of the pointers will cause null pointer exceptions.
    for( vector<circle*>::iterator i = v.begin();
            i != v.end(); ++i ) {
        delete *i; // not exception safe
    }

    // Don't forget to delete this, too.
    delete p;
} // end f()
```

>Here's how the same thing is accomplished in modern C++:

下面是如何在现代C++中实现同样的事情：

```
#include <memory>
#include <vector>

void f()
{
    // ...
    auto p = make_shared<circle>( 42 );
    vector<shared_ptr<shape>> v = load_shapes();

    for( auto& s : v )
    {
        if( s && *s == *p )
        {
            cout << *s << " is a match\n";
        }
    }
}
```

>In modern C++, you don't have to use new/delete or explicit exception handling because you can use smart pointers instead. When you use the **auto** type deduction and [lambda function](https://docs.microsoft.com/en-us/cpp/cpp/lambda-expressions-in-cpp?view=vs-2019), you can write code quicker, tighten it, and understand it better. And a range-based **for** loop is cleaner, easier to use, and less prone to unintended errors than a C-style **for** loop. You can use boilerplate together with minimal lines of code to write your app. And you can make that code exception-safe and memory-safe, and have no allocation/deallocation or error codes to deal with.

在现代C++中，你不必使用`new/delete`或显式的异常处理，因为你可以使用智能指针来代替。当您使用**auto**类型推导和lambda函数时，您可以更快地编写代码，更加精简，并更好地理解它。与C样式的**for**循环相比，基于范围的**for**循环更干净、更容易使用，并且不容易出现意外错误。您可以使用样板文件和最少的代码行来编写应用程序。并且您可以使代码异常安全和内存安全，并且没有要处理的分配/释放或错误代码

>Modern C++ incorporates two kinds of polymorphism: compile-time, through templates, and run-time, through inheritance and virtualization. You can mix the two kinds of polymorphism to great effect. The C++ Standard Library template `shared_ptr` uses internal virtual methods to accomplish its apparently effortless type erasure. But don't over-use virtualization for polymorphism when a template is the better choice. Templates can be very powerful.

现代C++将两种多态性结合在一起：编译时间通过继承，以及运行时间通过虚拟化。您可以将这两种多态性混合到一起，以获得巨大的效果。C++标准库模板`shared_ptr`使用内部虚拟方法来完成其显然毫不费力的类型擦除。但是当模板是更好的选择时，不要过度使用虚拟化来实现多态性。模板可能非常强大

>If you're coming to C++ from another language, especially from a managed language in which most of the types are reference types and very few are value types, know that C++ classes are value types by default. But you can specify them as reference types to enable polymorphic behavior that supports object-oriented programming. A helpful perspective: value types are more about memory and layout control, reference types are more about base classes and virtual functions to support polymorphism. By default, value types are copyable—they each have a copy constructor and a copy assignment operator. When you specify a reference type, make the class non-copyable—disable the copy constructor and copy assignment operator—and use a virtual destructor, which supports the polymorphism. Value types are also about the contents, which, when they are copied, give you two independent values that you can modify separately. But reference types are about identity—what kind of object it is—and for this reason are sometimes referred to as polymorphic types.

如果您从另一种语言进入C++，特别是从大多数类型是引用类型且很少有值类型的托管语言中，需要知道默认情况下C++类是值类型的。但您可以将它们指定为引用类型，以启用支持面向对象编程的多态行为。一个有用的视角：值类型更多地是关于内存和布局控制，引用类型更多地是关于基类和支持多态性的虚拟函数。默认情况下，值类型是可复制的，它们都有一个复制构造函数和一个复制分配运算符。指定引用类型时，请使类不可复制 -- 禁用复制构造函数和复制分配运算符，并使用支持多态性的虚拟析构函数。值类型也与内容有关，复制内容时，会为您提供两个独立的值，您可以分别修改这些值。但是引用类型是关于标识 -- 它是什么类型的对象，因此有时被称为多态类型

>C++ is experiencing a renaissance because power is king again. Languages like Java and C# are good when programmer productivity is important, but they show their limitations when power and performance are paramount. For high efficiency and power, especially on devices that have limited hardware, nothing beats modern C++.

C++正在经历复兴，因为能力再次成为国王。像Java和C#这样的语言在程序员的生产力很重要的时候是很好的，但是它们在功率和性能是最重要的时候显示出它们的局限性。对于高效率和功率，特别是在硬件有限的设备上，没有什么比现代C++更出色

>Not only the language is modern, the development tools are, too. Visual Studio makes all parts of the development cycle robust and efficient. It includes Application Lifecycle Management (ALM) tools, IDE enhancements like IntelliSense, tool-friendly mechanisms like XAML, and building, debugging, and many other tools.

不仅语言是现代的，开发工具也是。Visual Studio使开发周期的所有部分都具有健壮性和高效性。它包括应用程序生命周期管理（ALM）工具、诸如IntelliSense之类的IDE增强、诸如XAML之类的工具友好机制以及构建、调试和许多其他工具

>The articles in this part of the documentation provide high-level guidelines and best practices for the most important features and techniques for writing modern C++ programs.

文档的这一部分为编写现代C++程序最重要的特征和技术提供了高层次的指导和最佳实践。

>* [C++ Type System](https://docs.microsoft.com/en-us/cpp/cpp/cpp-type-system-modern-cpp?view=vs-2019)
>* [Uniform Initialization and Delegating Constructors](https://docs.microsoft.com/en-us/cpp/cpp/uniform-initialization-and-delegating-constructors?view=vs-2019)
>* [Object Lifetime And Resource Management](https://docs.microsoft.com/en-us/cpp/cpp/object-lifetime-and-resource-management-modern-cpp?view=vs-2019)
>* [Objects Own Resources (RAII)](https://docs.microsoft.com/en-us/cpp/cpp/objects-own-resources-raii?view=vs-2019)
>* [Smart Pointers](https://docs.microsoft.com/en-us/cpp/cpp/smart-pointers-modern-cpp?view=vs-2019)
>* [Pimpl For Compile-Time Encapsulation](https://docs.microsoft.com/en-us/cpp/cpp/pimpl-for-compile-time-encapsulation-modern-cpp?view=vs-2019)
>* [Containers](https://docs.microsoft.com/en-us/cpp/cpp/containers-modern-cpp?view=vs-2019)
>* [Algorithms](https://docs.microsoft.com/en-us/cpp/cpp/algorithms-modern-cpp?view=vs-2019)
>* [String and I/O Formatting (Modern C++)](https://docs.microsoft.com/en-us/cpp/cpp/string-and-i-o-formatting-modern-cpp?view=vs-2019)
>* [Errors and Exception Handling](https://docs.microsoft.com/en-us/cpp/cpp/errors-and-exception-handling-modern-cpp?view=vs-2019)
>* [Portability At ABI Boundaries](https://docs.microsoft.com/en-us/cpp/cpp/portability-at-abi-boundaries-modern-cpp?view=vs-2019)

* C++类型系统
* 统一初始化和委托构造器
* 对象生命周期和资源管理
* 对象拥有资源（RAII）
* 智能指针
* 用于编译时封装的Pimpl
* 容器
* 算法
* 字符串和I/O格式（现代C++）
* 错误和异常处理
* ABI边界的可移植性

>For more information, see the Stack Overflow article [Which C++ idioms are deprecated in C++11](https://stackoverflow.com/questions/9299101/which-c-idioms-are-deprecated-in-c11).

有关更多信息，请参见文章`Which C++ idioms are deprecated in C++11`