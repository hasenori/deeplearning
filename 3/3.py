# coding: utf-8
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

#画像テスト用データインポート
#--------------------------------
import sys, os
from dataset.mnist import load_mnist
#--------------------------------

#活性化関数一覧
#--------------------------------
#ステップ関数
def step_function(x):
    return np.array(x > 0, dtype=np.int)

#シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

#ReLu関数    
def relu(x):
    return np.maximum(0,x)

#恒等関数
def identity_function(x):
    return x

#ソフトマックス関数
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) #指数関数 cは、オーバーフロー対策
    sum_exp_a = np.sum(exp_a) #指数関数の和
    y = exp_a / sum_exp_a
    return y

a = np.array([10000.3,2.9,4.0])
y = softmax(a)
print(np.sum(y))


x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
y2 = step_function(x)
y3 = relu(x)

plt.plot(x, y1)
plt.plot(x, y2, 'k--')
plt.plot(x, y3)
plt.ylim(-0.1, 1.1) #図で描画するy軸の範囲を指定

#画像出力
#--------------------------------
plt.savefig('../../../../var/www/html/images/graph.png')

#--------------------------------
# 活性化関数による3層ニューラルネットワークのプロセス
#--------------------------------
def init_network():
    network = {}
    #ニューラルネットワークの慣例として、重みだけ大文字で表記
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])

    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network, x)
print(y)

#--------------------------------
# MINISTデータセット
#--------------------------------
# 最初の呼び出しは数分待ちます・・・
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)






