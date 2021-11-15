import numpy as np
import sys
sys.path.append('/Volumes/DevHD/LearningProjects/DeepLearning/deeplearning_from_scratch')
from dataset.mnist import load_mnist


(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # 원-핫 인코딩 된 정답 레이블 (60000, 10)

# 무작위 10개 추출
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print(f'x_batch : {x_batch}')
print(f't_batch : {t_batch}')

def cross_entropy_error_for_one(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y[np.arange(batch_size), t])) / batch_size

print(f't : {t_train[0]}')
print(f'x : {x_train[0]}')
print(cross_entropy_error_for_one(np.array([0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]), 
    np.array(t_train[0])))

