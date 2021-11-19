import numpy as np
import sys
sys.path.append('/Volumes/DevHD/LearningProjects/DeepLearning/deeplearning_from_scratch')
from dataset.mnist import load_mnist


def cross_entropy_error_for_one_hot_label(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    delta = 1e-7
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error_for_one_hot_label(np.array(y), np.array(t)))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error_for_one_hot_label(np.array(y), np.array(t)))

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=False)

# 무작위 10개 추출
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print(f'x_batch : {x_batch.shape}')
print(f't_batch : {t_batch.shape}')

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    print(f'y : {y.shape}')
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y[np.arange(batch_size), t]) + 1e-7) / batch_size

# cross_entropy_error(np.array([0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]), t_batch)


y = np.array([0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0])
y = y.reshape(1, y.size)
batch_size = y.shape[0]
print(f'y : {y}')
print(f'batch_size : {batch_size}')
print(f'{np.arange(batch_size)}')

print(y[np.arange(batch_size), 0])

    

