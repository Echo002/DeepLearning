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

current_x = 0.5         # the algorithm starts at x=0.5
learning_rate = 0.01    # step size multiplier
num_iterations = 70     # the number of times to train the function


# the derivative of the error function (x**4 = the power of 4 or x^4)
def slope_at_given_x_value(x):
    return 5 * x ** 4 - 6 * x ** 2


# Move X to the right or left depending on the slope of the error function
for i in range(num_iterations):
    previous_x = current_x
    current_x += -learning_rate * slope_at_given_x_value(previous_x)
    print(previous_x)

print("The local minimum occurs at %f" % current_x)