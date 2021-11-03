import sys
import os
import pickle
import numpy as np
sys.path.append('/Volumes/DevHD/LearningProjects/DeepLearning/deeplearning_from_scratch')
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("./ch03/sample_weight.pkl", 'rb') as f:
        # 학습된 가중치 매개변수가 담긴 파일
        # 학습 없이 바로 추론을 수행
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    print(f'x : {x.shape}')
    print(f'W1 : {W1.shape}')
    print(f'z1 : {z1.shape}')
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    print(f'W2 : {W2.shape}')
    print(f'z2 : {z1.shape}')
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    print(f'W3 : {W2.shape}')
    print(f'y : {y.shape}')

    return y


x, t = get_data()
print(f'x : {x[0]}')
print(f't : {t}')
network = init_network()
accuracy_cnt = 0

'''
print(f'len(x) : {len(x)}')
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)  # 확률이 가장 높은 원소의 인덱스를 얻는다.

    print(f'y : {y}', f' p : {p}') if i == 0 else None
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))  # Accuracy:0.9352
'''

batch_size = 100

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    # print(f'x_batch : {x_batch}', f'\ny_batch : {y_batch}') if i == 0 else None
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))  # Accuracy:0.9352