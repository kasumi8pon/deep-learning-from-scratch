import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.random.randn(1000, 100) # 1000 個のデータ
node_num = 100 # 各隠れ層のノード（ニューロンの数）
hidden_layer_size = 5 # 隠れ層が 5 層
activations = {} # ここにアクティベーションの結果を格納する

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # 0 と 1 に偏った分布になる。逆伝播で勾配の値が小さくなって勾配消失が起きる。
    # w = np.random.randn(node_num, node_num) * 1

    # 0.5 付近に集中する分布となる。
    # w = np.random.randn(node_num, node_num) * 0.01

    # 広がりを持った分布になり、シグモイド関数の表現力も制限されず効率的に学習が行える。
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num)

    z = np.dot(x, w)
    a = sigmoid(z)
    activations[i] = a

for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten(), 30, range=(0, 1))

plt.show()
