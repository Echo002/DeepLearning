#!/usr/bin/env python
#!-*-coding:utf-8 -*-
#!@Author:xugao
#         ┌─┐       ┌─┐
#      ┌──┘ ┴───────┘ ┴──┐
#      │                 │
#      │                 │
#      │    ＞  　　＜    │
#      │                 │
#      │  ....　⌒　....　│
#      │                 │
#      └───┐         ┌───┘
#          │         │
#          │         │
#          │         │
#          │         └──────────────┐
#          │                        │
#          │                        ├─┐
#          │                        ┌─┘
#          │                        │
#          └─┐  ┐  ┌───────┬──┐  ┌──┘
#            │ ─┤ ─┤       │ ─┤ ─┤
#            └──┴──┘       └──┴──┘
#                神兽保佑
#                BUG是不可能有BUG的!
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#生成整数型的属性
def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#生成字符型的属性
def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

mnist=input_data.read_data_sets(r"E:\DataSet\MNIST",dtype=tf.uint8,one_hot=True)

# 保存相关数据
images=mnist.train.images
labels=mnist.train.labels
pixels=images.shape[1]
num_examples=mnist.train.num_examples

filename=r"E:\DataSet\MNIST\demo.tfrecords"
writer=tf.python_io.TFRecordWriter(filename)

for index in range(num_examples):
    # 将图像矩阵转化成一个字符串
    image_raw = images[index].tostring()
    # 将一个样例转化成Example Protocol Buffer，并将所有的信息写入这个数据结构
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))

    # 将一个Example写入TFRecord文件中
    writer.write(example.SerializeToString())
    if index%1000 == 0:
        print("%d has done."%index)
writer.close()