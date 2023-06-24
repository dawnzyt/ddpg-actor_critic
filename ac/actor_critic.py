import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, n_state, n_action, hidden1, hidden2):
        super().__init__()
        # actor网络
        self.fc1 = nn.Linear(n_state, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.mu_layer = nn.Linear(hidden2, n_action)
        self.sigma_layer = nn.Linear(hidden2, n_action)
        self.distribution = torch.distributions.Normal

    def forward(self, state):
        """
        前向传播
        :param state:
        :return:
        """
        x = self.fc1(state)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        mu = 2 * torch.tanh(self.mu_layer(x))
        sigma = F.softplus(self.sigma_layer(x)) + 1e-5
        dist = self.distribution(mu, sigma)
        return dist


class Critic(nn.Module):
    def __init__(self, n_state, n_action, hidden1, hidden2):
        super().__init__()
        # critic网络
        self.fc1 = nn.Linear(n_state, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.V_layer = nn.Linear(hidden2, n_action)

    def forward(self, state):
        """
        前向传播
        :param state:
        :return:
        """
        x = self.fc1(state)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        V = self.V_layer(x)
        return V


class ActorCritic(nn.Module):
    def __init__(self, n_state, n_action, hidden1, hidden2):
        super(ActorCritic, self).__init__()
        # actor、critic共享的前馈网络
        self.fc1 = nn.Linear(n_state, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        # actor网络输出动作概率正态分布即均值和方差
        self.mu_layer = nn.Linear(hidden2, n_action)
        self.sigma_layer = nn.Linear(hidden2, n_action)
        # critic网络输出critic对状态的期望价值
        self.V_layer = nn.Linear(hidden2, n_action)
        self.distribution = torch.distributions.Normal  # 使用正态分布

    def forward(self, x):
        """

        :param x: state batch, shape = (batch_size, n_state)
        :return:
        """
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        # tanh将动作均值限制在[-1,1]之间
        mu = 2 * torch.tanh(self.mu_layer(x))  # multiply by 2 to increase the action space
        # softplus将动作方差限制在[0,inf]之间, softplus是relu的平滑版本
        # softplus(x) = ln(1+exp(x))
        sigma = F.softplus(self.sigma_layer(x)) + 1e-5  # avoid 0
        # 生成正态分布
        dist = self.distribution(mu.view(1, -1).data, sigma.view(1, -1).data)
        # critic网络输出期望价值
        V = self.V_layer(x)
        return dist, V
