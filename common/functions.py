# coding: utf-8
import numpy as np

def ok():
    print('ok!')

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
