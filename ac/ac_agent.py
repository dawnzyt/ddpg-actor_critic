import copy

import numpy as np
import torch
import torch.nn as nn

import utils.ounoise, utils.dataloger
from ac.actor_critic import ActorCritic, Actor, Critic


class ActorCriticAgent:
    def __init__(self, env, hidden1, hidden2, gamma, lr_actor=0.00001, lr_critic=0.0001, buf_size=1000000, sync_freq=100,
                 batch_size=64, exp_name='exp1', device='cuda'):
        # 初始化AC智能体
        self.env = env
        self.n_state = env.observation_space.shape[0]
        self.n_action = env.action_space.shape[0]
        self.device = device
        self.actor = Actor(self.n_state, self.n_action, hidden1, hidden2).to(device)
        self.critic = Critic(self.n_state, self.n_action, hidden1, hidden2).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        # self.ac = ActorCritic(self.n_state, self.n_action, hidden1, hidden2)

        # 初始化经验回放
        self.buf_size = buf_size
        self.buffer = np.zeros((self.buf_size, self.n_state * 2 + 3))
        self.bf_counter = 0
        self.learn_counter = 0
        self.sync_freq = sync_freq

        # 初始化噪声
        # why noise? mountain car continuous环境是一个比较困难的环境, 为了增加探索性, 采用噪声
        # 否则很难到达终点
        self.explore_mu = 0.2
        self.explore_theta = 0.15
        self.explore_sigma = 0.2
        self.noise = utils.ounoise.OUNoise(self.n_action, self.explore_mu, self.explore_theta, self.explore_sigma)

        # 初始化超参数
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.batch_size = batch_size
        self.exp_name = exp_name
        # self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=self.lr)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.mse_loss = nn.MSELoss()
        self.loger = utils.dataloger.DataLoger('./result/ac/' + self.exp_name)

    def save_transition(self, state, action, reward, nxt_state, done):
        """
        经验重放, 保存经验
        """
        idx = self.bf_counter % self.buf_size
        self.buffer[idx, :] = np.hstack((state, action, reward, nxt_state, done))
        self.bf_counter += 1

    def predict(self, state):
        """
        利用actor得到的分布去采样预测一个动作
        """
        state = torch.Tensor(state).to(self.device)
        dist = self.actor(state)
        action = dist.sample().cpu().numpy()
        return action

    def sample_act(self, state, noise_scale=1.0):
        """
        从actor网络输出的正态分布中采样一个动作
        :param noise_scale:
        :param state: shape = (n_state,)
        :return:
        """
        state = torch.Tensor(state).to(self.device)
        dist = self.actor(state)
        action = dist.sample().cpu().numpy()
        return action + self.noise.sample() * noise_scale

    def sample_batch(self):
        max_buffer = min(self.bf_counter, self.buf_size)  # 当前buffer内已存储的经验
        # 随机抽取batch_size个经验
        idx = np.random.choice(max_buffer, self.batch_size, replace=False)
        # idx = np.array([(self.bf_counter - 1) % self.buf_size])
        batch = self.buffer[idx, :]
        state = batch[:, :self.n_state]
        action = batch[:, self.n_state:self.n_state + 1]
        reward = batch[:, self.n_state + 1:self.n_state + 2]
        nxt_state = batch[:, self.n_state + 2:self.n_state * 2 + 2]
        done = batch[:, self.n_state * 2 + 2:]
        return state, action, reward, nxt_state, done

    def learn(self, epoch, step, cur_state):
        """
        更新网络参数
        :return:
        """
        max_buffer = min(self.bf_counter, self.buf_size)  # 当前buffer内已存储的历史经验回放数据
        if max_buffer < self.batch_size:
            return
        self.learn_counter += 1
        if self.learn_counter % self.sync_freq == 0:  # 每隔sync_freq步, 更新target_critic网络的参数
            self.target_critic.load_state_dict(self.critic.state_dict())
        # 从buffer中随机抽取batch_size个经验
        state, action, reward, nxt_state, done = self.sample_batch()

        state = torch.Tensor(state).to(self.device)
        action = torch.Tensor(action).to(self.device)
        reward = torch.Tensor(reward).to(self.device)
        nxt_state = torch.Tensor(nxt_state).to(self.device)
        done = torch.Tensor(done).to(self.device)

        dist, V = self.actor(state), self.critic(state)
        nxt_V = self.target_critic(nxt_state).detach()
        TD_error = reward + self.gamma * nxt_V * (1 - done) - V
        # actor网络的损失函数: 对于action做带权重的梯度更新
        # TD_error.detach()是为了防止梯度更新到critic网络,TD_error大, actor_loss大, actor网络的梯度更新大
        actor_loss = -dist.log_prob(action) * TD_error.detach()
        # critic网络的损失函数: TD_error的平方。
        # 目的
        critic_loss = self.mse_loss(V, reward + self.gamma * nxt_V * (1 - done))
        # loss = actor_loss + critic_loss
        # loss = loss.mean()
        actor_loss = actor_loss.mean()

        # loss backward and update
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        self.optim_actor.zero_grad()
        actor_loss.backward()
        # 梯度裁剪
        nn.utils.clip_grad_norm_(self.actor.parameters(), 3)
        self.optim_actor.step()
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()

        if step % 100 == 0:
            self.loger.log('actor_loss', actor_loss.item(), step)
            self.loger.log('critic_loss', critic_loss.item(), step)
            if step % 1000 == 0:
                print('epoch: {}, step: {}, actor_loss: {}, critic_loss: {}, TD_error:{}'.format(
                    epoch, step, actor_loss, critic_loss, TD_error.mean()))

    def save_model(self, epoch, avg_score):
        torch.save(self.actor.state_dict(),
                   './result/ac/' + self.exp_name + '/actor_epoch{}_avgScore{:.3f}.pth'.format(epoch, avg_score))
        torch.save(self.critic.state_dict(),
                   './result/ac/' + self.exp_name + '/critic_epoch{}_avgScore{:.3f}.pth'.format(epoch, avg_score))

    def load_model(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
