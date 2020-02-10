---
title: tf.Variable() 和tf.get_variable()
date: 2019-08-11 17:46:26
tags: TensorFlow
categories: TensorFlow
---
# tf.Variable() 和tf.get_variable()的区别以及tf.variable_scope的使用
<!--more-->
### 在tensorflow中提供了tf.get_variable函数来创建或者获取变量。当tf.get_variable用于创建变量时，则与tf.Variable的功能基本相同。
```
#定义的基本等价
v = tf.get_variable("v",shape=[1],initializer.constant_initializer(1.0))
v = tf.Variable(tf.constant(1.0,shape=[1]),name="v")
```
### 不同点：1.使用tf.Variable时，如果检测到命名冲突，系统会自己处理。使用tf.get_variable()时，系统不会处理冲突，而会报错   2.两函数指定变量名称的参数不同，对于tf.Variable函数，变量名称是一个可选的参数，通过name="v"的形式给出,而tf.get_variable函数，变量名称是一个必填的参数，它会根据变量名称去创建或者获取变量
### 先通过tf.variable_scope生成一个上下文管理器，并指明需求的变量在这个上下文管理器中，就可以直接通过tf.get_variable获取已经生成的变量。
```
#通过tf.variable_scope函数控制tf.get_variable函数来获取以及创建过的变量
with tf.variable_scope("zyy"):#zyy的命名空间
        v=tf.get_variable("v",[1],initializer=tf.constant_initializer(1.0))  #在zyy的命名空间内创建名字为v的变量
```
```
with tf.variable_scope("zyy"):
         v=tf.get_variable("v",[1])  #通过tf.get_variable函数创建v的变量，则会失败，由于在zyy空间中已经生成了一个v的变量
```
### 在上下文管理器中已经生成一个v的变量，若想通过tf.get_variable函数获取其变量，则可以通过reuse参数的设定为True来获取（可以将reuse按照字面意思理解，重用）
```
with tf.variable_scope("zyy",reuse=True):
      v1=tf.get_variable("v",[1])
print v==v1   #输出为True
```
### 并且tf.variable_scope只能获取已经创建过的变量。
### 如果tf.variable_scope函数使用参数reuse=None或者reuse=False创建上下文管理器，则tf.get_variable函数可以创建新的变量。但不可以创建已经存在的变量即为同名的变量。