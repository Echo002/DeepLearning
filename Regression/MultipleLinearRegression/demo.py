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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
data = pd.read_csv('demo.csv',encoding='GBK')
# print(data)
font={'family':'SimHei'}
matplotlib.rc('font',**font)
pd.plotting.scatter_matrix(data[['percent','brokerage','sell']], alpha=0.7, figsize=(14,8), diagonal='hist')
# plt.show()
# print(data[["percent","brokerage","sell"]].corr())
x = data[["percent","brokerage"]]
y = data[["sell"]]

demoModel = LinearRegression()
demoModel.fit(x,y)

print(demoModel.coef_)
print(demoModel.intercept_)
demoModel.predict([11,50])
'''
# github测试文件
# 再次测试
#建模
#训练模型
'''






