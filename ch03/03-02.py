import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    # y = x > 0
    # return y.astype(int)
    return np.array(x > 0, dtype=int)


# x = np.arange(-5.0, 5.0, 0.1)
# print(x)
# y = step_function(x)
# print(y)

# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)  
# plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
# x = [n for n in range(-5, 6, 1)]
print(x)

y = sigmoid(x)
print(y)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)  
plt.show()


def relu(x):
    return np.maximum(0, x)