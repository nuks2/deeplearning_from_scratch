import numpy as np


x = np.array([1.0, 2.0, 3.0])
print(x)
print(type(x))

y = np.array([2.0, 4.0, 6.0])

# print(x + y)
# print(x - y)
# print(x * y)
# print(x / y)
# print(x / 2.0)
# print('----------------------')

A = np.array([[1, 2], [3, 4]])
B = np.array([[3, 0], [0, 6]])

# print(A)
# print(B)
# print(A.shape)
# print(A.dtype)

# print(A + B)
# print(A * B)
# print(A * 10)

X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)

X = X.flatten()  
print(X)  
print(X[np.array([0, 2, 4])])  # 인덱스가 0,2,4인 원소 얻기 [51 14  0]
print(X > 15)  
print(X[X > 15])  


