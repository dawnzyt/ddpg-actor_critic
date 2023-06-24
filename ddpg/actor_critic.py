import torch
import torch.nn as nn
import torch.nn.functional as F


# DDPG的actor网络: 输入state，输出使得输入到critic网络估计价值最大的action
class DDPGActor(nn.Module):
    def __init__(self, n_state, n_action, hidden1, hidden2):
        super().__init__()
        n_action = 1
        self.fnn = nn.Sequential(nn.Linear(n_state, hidden1),
                                 # nn.BatchNorm1d(hidden1),
                                 nn.ReLU(),
                                 nn.Linear(hidden1, hidden2),
                                 # nn.BatchNorm1d(hidden2),
                                 nn.ReLU(),
                                 nn.Linear(hidden2, n_action),
                                 nn.Tanh())

    def forward(self, inputs):
        """
        :param inputs: shape: [batch_size, n_state]
        :return: action: shape: [batch_size, n_action]
        """
        return self.fnn(inputs)


# DDPG的critic网络: 输入state和action，输出Q(s,a)
class DDPGCritic(nn.Module):
    def __init__(self, n_state, n_action, hidden1, hidden2):
        super().__init__()
        # critic网络: 两路输入，分别是state和action
        self.fnn_state = nn.Sequential(nn.Linear(n_state, hidden1),
                                       # nn.BatchNorm1d(hidden1),
                                       nn.ReLU(),
                                       nn.Linear(hidden1, hidden2))
        self.fnn_action = nn.Sequential(nn.Linear(n_action, hidden1),
                                        nn.ReLU(),
                                        nn.Linear(hidden1, hidden2))
        # 将两路相加
        # 激活
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden2, 1)

    def forward(self, state, action):
        """
        :param state: shape: [batch_size, n_state]
        :param action: shape: [batch_size, n_action]
        :return: Q(s,a): shape: [batch_size, 1]
        """
        x_state = self.fnn_state(state)
        x_action = self.fnn_action(action)
        x = x_state + x_action
        x = self.relu(x)
        return self.output_layer(x)
