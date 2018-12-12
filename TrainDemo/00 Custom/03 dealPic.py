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

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

image_raw_data = tf.gfile.FastGFile("timg.jpg", 'rb').read()

# 图像的编码
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_raw_data)
#
#     # 输出解码之后的三维矩阵。
#     print(img_data.eval())
#     plt.imshow(img_data.eval())
#     plt.show()
#
#     encode_image = tf.image.encode_jpeg(img_data)
#     with tf.gfile.GFile(r"E:\Code\DeepLearning\TrainDemo\00 Custom\output.jpeg","wb") as f:
#         f.write(encode_image.eval())

# 图像大小的调整
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     # 如果直接以0-255范围的整数数据输入resize_images，那么输出将是0-255之间的实数，
#     # 不利于后续处理。本书建议在调整图片大小前，先将图片转为0-1范围的实数。
#     image_float = tf.image.convert_image_dtype(img_data, tf.float32)
#     resized = tf.image.resize_images(image_float, [27, 27], method=0)
#
#     plt.imshow(resized.eval())
#     plt.show()

# 图像的裁剪
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     croped = tf.image.resize_image_with_crop_or_pad(img_data, 100, 100)
#     padded = tf.image.resize_image_with_crop_or_pad(img_data, 2000, 2000)
#     plt.imshow(croped.eval())
#     plt.show()
#     plt.imshow(padded.eval())
#     plt.show()

# 图像的翻转
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     # 上下翻转
#     # flipped1 = tf.image.flip_up_down(img_data)
#     # 左右翻转
#     # flipped2 = tf.image.flip_left_right(img_data)
#
#     # 对角线翻转
#     transposed = tf.image.transpose_image(img_data)
#     plt.imshow(transposed.eval())
#     plt.show()
#     # 以一定概率上下翻转图片。
#     # flipped = tf.image.random_flip_up_down(img_data)
#     # 以一定概率左右翻转图片。
#     # flipped = tf.image.random_flip_left_right(img_data)

# 图像的色彩调整
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     # 在进行一系列图片调整前，先将图片转换为实数形式，有利于保持计算精度。
#     image_float = tf.image.convert_image_dtype(img_data, tf.float32)
#
#     # 将图片的亮度-0.5。
#     # adjusted = tf.image.adjust_brightness(image_float, -0.5)
#
#     # 将图片的亮度-0.5
#     # adjusted = tf.image.adjust_brightness(image_float, 0.5)
#
#     # 在[-max_delta, max_delta)的范围随机调整图片的亮度。
#     adjusted = tf.image.random_brightness(image_float, max_delta=0.5)
#
#     # 将图片的对比度-5
#     # adjusted = tf.image.adjust_contrast(image_float, -5)
#
#     # 将图片的对比度+5
#     adjusted = tf.image.adjust_contrast(image_float, 10)
#
#     # 在[lower, upper]的范围随机调整图的对比度。
#     # adjusted = tf.image.random_contrast(image_float, lower, upper)
#
#     # 在最终输出前，将实数取值截取到0-1范围内。
#     adjusted = tf.clip_by_value(adjusted, 0.0, 1.0)
#     plt.imshow(adjusted.eval())
#     plt.show()

# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     # 在进行一系列图片调整前，先将图片转换为实数形式，有利于保持计算精度。
#     image_float = tf.image.convert_image_dtype(img_data, tf.float32)
#
#     adjusted = tf.image.adjust_hue(image_float, 0.1)
#     # adjusted = tf.image.adjust_hue(image_float, 0.3)
#     # adjusted = tf.image.adjust_hue(image_float, 0.6)
#     # adjusted = tf.image.adjust_hue(image_float, 0.9)
#
#     # 在[-max_delta, max_delta]的范围随机调整图片的色相。max_delta的取值在[0, 0.5]之间。
#     # adjusted = tf.image.random_hue(image_float, max_delta)
#
#     # 将图片的饱和度-5。
#     adjusted = tf.image.adjust_saturation(image_float, -5)
#     # 将图片的饱和度+5。
#     # adjusted = tf.image.adjust_saturation(image_float, 5)
#     # 在[lower, upper]的范围随机调整图的饱和度。
#     # adjusted = tf.image.random_saturation(image_float, lower, upper)
#
#     # 将代表一张图片的三维矩阵中的数字均值变为0，方差变为1。
#     # adjusted = tf.image.per_image_whitening(image_float)
#
#     # 在最终输出前，将实数取值截取到0-1范围内。
#     adjusted = tf.clip_by_value(adjusted, 0.0, 1.0)
#     plt.imshow(adjusted.eval())
#     plt.show()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])

    # sample_distorted_bounding_box要求输入图片必须是实数类型。
    image_float = tf.image.convert_image_dtype(img_data, tf.float32)

    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(image_float), bounding_boxes=boxes, min_object_covered=0.4)

    # 截取后的图片
    distorted_image = tf.slice(image_float, begin, size)
    plt.imshow(distorted_image.eval())
    plt.show()

    # 在原图上用标注框画出截取的范围。由于原图的分辨率较大（2673x1797)，生成的标注框
    # 在Jupyter Notebook上通常因边框过细而无法分辨，这里为了演示方便先缩小分辨率。
    image_small = tf.image.resize_images(image_float, [180, 267], method=0)
    batchced_img = tf.expand_dims(image_small, 0)
    image_with_box = tf.image.draw_bounding_boxes(batchced_img, bbox_for_draw)
    print(bbox_for_draw.eval())
    plt.imshow(image_with_box[0].eval())
    plt.show()