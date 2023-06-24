import sys

import torch
import tqdm

import utils.utils
from ac.ac_agent import ActorCriticAgent
from ddpg.ddpg_agent import DDPGAgent
import gym
import numpy as np

exp_name = 'exp2'
agent_name = 'DDPG'
if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    env = env.unwrapped  # 取消限制
    # 定义agent
    if agent_name == 'AC':
        agent = ActorCriticAgent(env, hidden1=256, hidden2=256, gamma=0.99, lr_actor=0.00001, lr_critic=0.0001,
                                 exp_name=exp_name)
    elif agent_name == 'DDPG':
        agent = DDPGAgent(env, hidden1=256, hidden2=256, gamma=0.99, lr_actor=0.0001, lr_critic=0.001,
                          exp_name=exp_name)
    episodes = 1000
    scores, avg_scores, turns, goals = [], [], [], []  # 记录每个episode的分数，平均分数，回合数，目标分数
    step = 0
    for episode in tqdm.tqdm(range(episodes), file=sys.stdout):
        state = env.reset()[0]
        score = 0
        while True:
            step += 1
            action = agent.sample_act(state)
            nxt_state, reward, done, _, info = env.step(action)
            score += reward
            # save
            agent.save_transition(state, action, reward, nxt_state, done)
            # learn
            agent.learn(episode, step, state)
            state = nxt_state
            if done or _:  # terminated
                break
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        turns.append(episode)
        goals.append(90)
        print("Episode {0}/{1}, Score: {2}, AVG Score: {3}".format(episode, episodes, score, avg_score))
        # 保存模型
        if avg_score >= 80:
            agent.save_model(episode, avg_score)
    utils.utils.plt_graph(turns, scores, avg_scores, goals, 'MountainCarContinuous-v0', agent_name, 'TRAIN',
                          './result/{}/{}'.format(agent_name, exp_name))
