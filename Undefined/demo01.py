import numpy as np

x_train = np.array([[1,2,3],[4,5,6]])
# m = len(x_train)
# result = np.zeros((m,10))
# print(result)
# for i,j in enumerate(x_train):
#     print(i,j)
#     result[i,j] = 1
# print(result)

x_train[0, [1,2]] = 1
print(x_train)