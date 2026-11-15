"""通用训练循环 Trainer，第12讲、第14讲、第15讲用。

改编自《深度学习入门：基于Python的理论与实现》官方源码 common/trainer.py
（MIT 许可，Copyright (c) 2016 Koki Saitoh）。构造参数、属性名（train_loss_list、train_acc_list、
test_acc_list）和方法名（train_step、train）与原书一致；注释改写成中文，另加了 max_iter、seed 两个可选参数。

每一步：随机抽一个 mini-batch → network.gradient → optimizer.update；
每过一个 epoch 在训练集、测试集上各评一次精度，记进列表，方便画曲线。
"""
import numpy as np

from common.optimizer import make_optimizer


class Trainer:
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100, optimizer="SGD", optimizer_param=None,
                 evaluate_sample_num_per_epoch=None, verbose=True, max_iter=None, seed=None):
        self.network = network
        self.x_train, self.t_train = x_train, t_train
        self.x_test, self.t_test = x_test, t_test
        self.batch_size = mini_batch_size
        self.optimizer = make_optimizer(optimizer, **(optimizer_param or {"lr": 0.01}))
        self.eval_num = evaluate_sample_num_per_epoch
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        n = x_train.shape[0]
        self.iter_per_epoch = max(n // mini_batch_size, 1)
        self.max_iter = max_iter if max_iter is not None else epochs * self.iter_per_epoch
        self.current_iter = 0
        self.current_epoch = 0
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def _acc(self, x, t):
        if self.eval_num:
            x, t = x[:self.eval_num], t[:self.eval_num]
        return self.network.accuracy(x, t)

    def train_step(self):
        pick = self.rng.choice(self.x_train.shape[0], self.batch_size, replace=False)
        xb, tb = self.x_train[pick], self.t_train[pick]
        grads = self.network.gradient(xb, tb)
        self.optimizer.update(self.network.params, grads)
        loss = self.network.loss(xb, tb)
        self.train_loss_list.append(loss)
        self.current_iter += 1
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            tr = self._acc(self.x_train, self.t_train)
            te = self._acc(self.x_test, self.t_test)
            self.train_acc_list.append(tr)
            self.test_acc_list.append(te)
            if self.verbose:
                print(f"epoch {self.current_epoch}: 训练精度 {tr:.4f}  测试精度 {te:.4f}  损失 {loss:.4f}")
        return loss

    def train(self):
        while self.current_iter < self.max_iter:
            self.train_step()
        final = self.network.accuracy(self.x_test, self.t_test)
        if self.verbose:
            print(f"最终测试精度 {final:.4f}")
        return final
