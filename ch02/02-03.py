import numpy as np


# 2.3.1 간단한 구현부터
def simple_and(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    if (x1 * w1 + x2 * w2) <= theta:
        return 0
    else:
        return 1

print(simple_and(1, 0.5))

# 2.3.2 가중치와 편향 도입
x = np.array([0, 1])  # 입력
w = np.array([0.5, 0.5])  # 가중치
b = -0.7  # 편향

print(w * x)
print(np.sum(w * x))        # 0.5
print(np.sum(w * x) + b)    # -0.2

# 2.3.3 가중치와 편향 구현하기
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    if (np.sum(w * x) + b) <= 0:
        return 0
    else:
        return 1

print("AND")
print(AND(0, 0))  # 0
print(AND(0, 1))  # 0
print(AND(1, 0))  # 0
print(AND(1, 1))  # 1