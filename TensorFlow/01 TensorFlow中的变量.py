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
'''
图解笔记完成情况
5 6 7 8 9 10 14
'''
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 创建了两个常量
m1 = tf.constant([[3,3]])
m2 = tf.constant([[2],[3]])

# 创建一个矩阵乘法，把m1和m2传入
product = tf.matmul(m1,m2)
print(product)

# # 定义一个会话，启动默认图
# sess = tf.Session()
# # 调用run方法执行矩阵乘法 会触发三个op
# result = sess.run(product)
# print(result)
# sess.close()

x = tf.Variable([1,2])
a = tf.constant([3,3])

sub = tf.subtract(x,a)
add = tf.add(x,sub)

init = tf.global_variables_initializer()

with tf.Session() as sess_2:
    sess_2.run(init)
    # result = sess_2.run(product)
    print(sess_2.run(sub))
    print(sess_2.run(add))

state = tf.Variable(0,name='counter')
# 创建一个变量初始化为0
new_value = tf.add(state , 1)
# 创建一个op使得state的值+1
update = tf.assign(state , new_value)
# 赋值op
init = tf.global_variables_initializer()
# 变量初始化
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _  in range(5):
        sess.run(update)
        print(sess.run(state))