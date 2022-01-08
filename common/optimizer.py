import numpy as np

# 確率的勾配降下法 Stochastic Gradient Descent
class SGD:
    def __init__(self, lr=0.01):
      self.lr = lr

    def update(self, params , grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None :
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]

class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
          self.h = {}
          for key, val in params.items():
              self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key] # 勾配の二乗和
            # 学習のスケールを調整する。1e-7 を加算して、0 でする除算ことを防ぐ
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key] + 1e-7))s
