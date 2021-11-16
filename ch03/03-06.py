import sys
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
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
print(f'x Shape : {x.shape}')
print(f't Shape : {t.shape}')

W1, W2, W3 = network['W1'], network['W2'], network['W3']
# print(f'W1 Shape : {W1.shape}')
# print(f'W1 : {W1}')
# print(f'W2 Shape : {W2.shape}')
# print(f'W2 : {W2}')
# print(f'W3 Shape : {W3.shape}')
# print(f'W3 : {W3}')
print(f'x len : {len(x)}')

for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)  # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))  # Accuracy:0.9352


batch_size = 10
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    
    p = np.argmax(y_batch, axis=1)
    print(f'p : {p}')
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))  # Accuracy:0.9352
