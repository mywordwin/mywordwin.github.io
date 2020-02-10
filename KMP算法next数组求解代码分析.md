---
title: KMP算法next数组求解代码分析
date: 2020-02-10 13:06:24
tags: KMP算法
categories: 算法
---
## 在看KMP算法时，对next数组求解代码有点迷，找了些资料才理解，在这做一下笔记
<!--more-->
### 这里就只说一下next数组的算法，其他的比较好理解，就不细说了
```c
int GetNext(char ch[],int cLen,int next[]){//cLen为串ch的长度
    next[1] = 0;
    int j = 1,k = 0;
    while(j<=cLen){
        if(k==0||ch[j]==ch[k]) next[++j] = ++k;
        else k = next[k];
    }
}
```
### 就这几行代码把我整迷糊了好长时间，先说一下next数组的意义：当主串和子串不匹配时，i不变，为了表示下一轮比较j定位的地方，我们将其定义为next[j]，next[j]就是第j个元素前j-1个元素首尾重合部分个数加一，为了能遍历完整，首尾重合部分的元素个数应取到最多，即next[j]应取尽量大的值

### 公式如下
![](KMP算法next数组求解代码分析\1.jpg)
### 再进一步想，next值是一个“工具”，我们单独的求next[j+1]是完全没有意义的，就是说要求next就要把所有j的next求出来。所有一般的，我们都是已知前j个元素的next值，求next[j+1]，以此递推下去，求完整的next数组。
```
next[j+1]的最大值为next[j]+1。
因为：
假设next[j]=k1，则可以说明P1…Pk1-1=Pj-k1+1…Pj-1，且这是前j个元素最大的首尾重合序列。
如果Pk1=Pj，那么P1…Pk1-1PK=Pj-k1+1…Pj-1Pj，那么k+1这也是前j+1个元素的最大首尾重合序列，也即next[j+1]的值为k1+1
如果Pk1≠Pj，那么next[j+1]可能的次大值为next[next[j]]+1，以此类推即可高效求出next[j+1]
```
### 这是自己理解的 
![](KMP算法next数组求解代码分析\2.jpg)
### 再上几张别人的图
![](KMP算法next数组求解代码分析\3.jpg)