import sys

import torch
import tqdm

import utils.utils
from ac.ac_agent import ActorCriticAgent
from ddpg.ddpg_agent import DDPGAgent
import gym
import numpy as np

exp_name = 'exp1'
agent_name = 'DDPG'
if_render = False
if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0', render_mode='human' if if_render else 'rgb_array')
    env = env.unwrapped
    # 定义agent
    if agent_name == 'AC':
        agent = ActorCriticAgent(env, hidden1=256, hidden2=256, gamma=0.99, lr_actor=0.00001, lr_critic=0.0001,
                                 exp_name=exp_name)
        agent.load_model(actor_path='result/ac/exp1/actor_epoch234_avgScore81.834.pth',
                         critic_path='result/ac/exp1/critic_epoch234_avgScore81.834.pth')
    elif agent_name == 'DDPG':
        agent = DDPGAgent(env, hidden1=256, hidden2=256, gamma=0.99, lr_actor=0.0001, lr_critic=0.001,
                          exp_name=exp_name)
        agent.load_model(actor_path='./result/ddpg/exp1/actor_epoch878_avgScore90.220.pth',
                         critic_path='./result/ddpg/exp1/critic_epoch878_avgScore90.220.pth')
    print(agent.actor)
    print(agent.critic)

    episodes = 100
    scores, avg_scores, turns, goals, steps, forces = [], [], [], [], [], []
    for episode in tqdm.tqdm(range(episodes), file=sys.stdout):
        state = env.reset()[0]
        score = 0
        step = 0
        force = 0
        while True:
            env.render()
            action = agent.predict(state)
            force += min(abs(action), 1)
            nxt_state, reward, done, _, info = env.step(action)
            score += reward
            step += 1
            state = nxt_state
            if done or _:
                break
        forces.append(force / step)
        steps.append(step)
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        turns.append(episode)
        goals.append(90)
        print("Episode {0}/{1}, Score: {2:.3f}, AVG Score: {3:.3f}, STEP: {4}, AVG Force: {5}".format(
            episode, episodes, score, avg_score, step, force / step))
    utils.utils.plt_graph(turns, scores, avg_scores, goals, 'MountainCarContinuous-v0', agent_name, 'TEST')
    print('avg_score:{}, avg_step:{}, avg_force:{}'.format(
        np.mean(scores[-100:]), np.mean(steps[-100:]), np.mean(forces[-100:])))
