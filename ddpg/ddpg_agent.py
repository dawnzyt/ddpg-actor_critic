import copy

import numpy as np
import torch

import utils.ounoise, utils.dataloger
from ddpg.actor_critic import DDPGActor, DDPGCritic


class DDPGAgent:

    def __init__(self, env, hidden1, hidden2, gamma, lr_actor=0.001, lr_critic=0.001, buf_size=1000000, sync_freq=100,
                 batch_size=64, exp_name='exp1', device='cuda'):
        # 初始化ddpg智能体
        self.env = env
        self.n_state = env.observation_space.shape[0]
        self.n_action = env.action_space.shape[0]
        self.device = device
        self.actor = DDPGActor(self.n_state, self.n_action, hidden1, hidden2).to(device)
        self.critic = DDPGCritic(self.n_state, self.n_action, hidden1, hidden2).to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        # 初始化噪声
        # why noise? mountain car continuous环境是一个比较困难的环境, 为了增加探索性, 采用噪声
        # 否则很难到达终点
        self.explore_mu = 0.2
        self.explore_theta = 0.15
        self.explore_sigma = 0.2
        self.noise = utils.ounoise.OUNoise(self.n_action, self.explore_mu, self.explore_theta, self.explore_sigma)

        # 初始化经验回放
        self.buf_size = buf_size
        self.buffer = np.zeros((self.buf_size, self.n_state * 2 + 3))
        self.bf_counter = 0
        self.learn_counter = 0
        self.sync_freq = sync_freq

        # 初始化超参数
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.batch_size = batch_size
        self.exp_name = exp_name
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.mse_loss = torch.nn.MSELoss()
        self.loger = utils.dataloger.DataLoger('./result/ddpg/' + self.exp_name)

    def learn(self, epoch, step, cur_state):
        """
        智能体学习, 更新actor和critic网络参数
        对于actor，更新目标是: 最大化Q(s,actor(s)), 即让actor网络输出的动作能够使Q值最大化，actor网络的损失函数: actor选择动作的负价值。对于critic，更新目标是: 建模(状态, 动作)->价值Q(s,a)，[网络输出预测最优价值]Q(s,a)---->逼近---->[target-实际回报]r + gamma * Q(s',a')。
        :param epoch:
        :param step:
        :return:
        """
        max_buffer = min(self.bf_counter, self.buf_size)  # 当前buffer内已存储的历史经验回放数据
        if max_buffer < self.batch_size:
            return
        self.learn_counter += 1
        if self.learn_counter % self.sync_freq == 0:  # 每隔一定步数同步一次target网络
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic.load_state_dict(self.critic.state_dict())

        # 从经验回放中随机抽取batch_size个经验
        state, action, reward, nxt_state, done = self.sample_batch()
        state = torch.Tensor(state).to(self.device)
        action = torch.Tensor(action).to(self.device)
        reward = torch.Tensor(reward).to(self.device)
        nxt_state = torch.Tensor(nxt_state).to(self.device)
        done = torch.Tensor(done).to(self.device)

        # 我们对critic的更新目标是: 建模(状态, 动作)->价值Q(s,a)
        # [网络输出预测最优价值]Q(s,a)---->逼近---->[target-实际回报]r + gamma * Q(s',a')

        # 计算target Q值
        nxt_action = self.target_actor(nxt_state)
        nxt_q = self.target_critic(nxt_state, nxt_action)
        target_q = reward + self.gamma * nxt_q * (1 - done)
        target_q = target_q.detach()  # target_q不需要梯度
        # 计算网络估计Q(s,a)
        q = self.critic(state, action)
        # 计算critic网络的损失函数:
        critic_loss = self.mse_loss(q, target_q)
        # actor更新目标是: 最大化Q(s,actor(s)), 即让actor网络输出的动作能够使Q值最大化
        # actor网络的损失函数: actor选择动作的负价值
        # actor_loss = -Q(s,actor(s))
        actor_loss = -self.target_critic(state, self.actor(state)).mean()

        # 1. 更新critic网络参数
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()
        # 2. 更新actor网络参数
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()

        # log
        if step % 100 == 0:
            self.loger.log('actor_loss', actor_loss.item(), step)
            self.loger.log('critic_loss', critic_loss.item(), step)
            if step % 1000 == 0:
                print('epoch: {}, step: {}, actor_loss: {}, critic_loss: {}, '
                      'cur_state:[{},{}]'.format(epoch, step, actor_loss, critic_loss, cur_state[0], cur_state[1]))

    def predict(self, state):
        """
        利用actor预测动作
        """
        state = torch.Tensor(state).to(self.device).view(1, -1)
        action = self.actor(state).cpu().detach().numpy()[0]
        return action

    def save_transition(self, state, action, reward, nxt_state, done):
        """
        经验重放, 保存经验
        """
        idx = self.bf_counter % self.buf_size
        self.buffer[idx, :] = np.hstack((state, action, reward, nxt_state, done))
        self.bf_counter += 1

    def sample_act(self, state, noise_scale=1.0):
        """
        从actor网络输出的正态分布中采样一个动作, 并加入噪声
        :param noise_scale:
        :param state: shape = (n_state,)
        :return:
        """
        state = torch.Tensor(state).to(self.device).view(1, -1)
        action = self.actor(state).cpu().detach().numpy()[0]
        return action + self.noise.sample() * noise_scale  # 加入噪声

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

    def save_model(self, epoch, avg_score):
        torch.save(self.actor.state_dict(),
                   './result/ddpg/' + self.exp_name + '/actor_epoch{}_avgScore{:.3f}.pth'.format(epoch, avg_score))
        torch.save(self.critic.state_dict(),
                   './result/ddpg/' + self.exp_name + '/critic_epoch{}_avgScore{:.3f}.pth'.format(epoch, avg_score))

    def load_model(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
