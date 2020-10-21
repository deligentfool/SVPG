import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from net import model
from svpg import _square_dist, _Kxx_dxKxx, calc_returns
from utils import vector_to_parameters, parameters_to_vector
import gym


class svpg_reinforce(object):
    def __init__(self, envs, gamma, learning_rate, episode, render, temperature, max_episode_length=1000):
        self.envs = envs
        self.num_agent = len(self.envs)
        self.observation_dim = self.envs[0].observation_space.shape[0]
        self.action_dim = self.envs[0].action_space.n
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.episode = episode
        self.render = render
        self.temperature = temperature
        self.eps = np.finfo(np.float32).eps.item()
        self.policies = [model(self.observation_dim, self.action_dim) for _ in range(self.num_agent)]
        self.optimizers = [torch.optim.Adam(self.policies[i].parameters(), lr=self.learning_rate) for i in range(self.num_agent)]
        self.total_returns = []
        self.weight_reward = None
        self.max_episode_length = max_episode_length

    #def train(self, ):
    #    total_returns = torch.FloatTensor(self.total_returns)
    #    eps = np.finfo(np.float32).eps.item()
    #    total_returns = (total_returns - total_returns.mean()) / (total_returns.std() + eps)
    #    log_probs = torch.cat(self.net.log_probs, 0)
    #    loss = (- log_probs * total_returns.detach())
    #    loss = loss.sum()
    #    self.writer.add_scalar('loss', loss, self.count)
    #    self.optimizer.zero_grad()
    #    loss.backward()
    #    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.1)
    #    self.optimizer.step()

    def train(self):
        policy_grads = []
        parameters = []

        for i in range(self.num_agent):
            agent_policy_grad = []
            returns = calc_returns(self.policies[i].rewards, self.gamma)
            returns = torch.FloatTensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + self.eps)

            for log_prob, r in zip(self.policies[i].log_probs, returns):
                agent_policy_grad.append(log_prob * r)

            self.optimizers[i].zero_grad()

            policy_grad = torch.cat(agent_policy_grad).sum()
            policy_grad.backward()

            param_vector, grad_vector = parameters_to_vector(self.policies[i].parameters(), both=True)
            policy_grads.append(grad_vector.unsqueeze(0))
            parameters.append(param_vector.unsqueeze(0))

        parameters = torch.cat(parameters)
        Kxx, dxKxx = _Kxx_dxKxx(parameters, self.num_agent)
        policy_grads = 1. / self.temperature * torch.cat(policy_grads)
        grad_logp = torch.mm(Kxx, policy_grads)
        grad_theta = - (grad_logp + dxKxx) / self.num_agent

        for i in range(self.num_agent):
            vector_to_parameters(grad_theta[i], self.policies[i].parameters(), grad=True)
            self.optimizers[i].step()
            del self.policies[i].rewards[:]
            del self.policies[i].log_probs[:]


    def run(self):
        for i_episode in range(self.episode):
            max_reward = -np.inf
            for i, env in enumerate(self.envs):
                obs = env.reset()
                total_reward = 0
                count = 0
                if self.render:
                    env.render()
                while count < self.max_episode_length:
                    action = self.policies[i].act(torch.FloatTensor(np.expand_dims(obs, 0)))
                    next_obs, reward, done, info = env.step(action)
                    self.policies[i].rewards.append(reward)
                    total_reward += reward
                    count += 1
                    if self.render:
                        env.render()
                    obs = next_obs
                    if done:
                        break
                if max_reward < total_reward:
                    max_reward = total_reward
            if self.weight_reward is None:
                self.weight_reward = max_reward
            else:
                self.weight_reward = 0.99 * self.weight_reward + 0.01 * max_reward
            print('episode: {}\t max_reward: {:.1f}\t weight_reward: {:.2f}'.format(i_episode + 1, max_reward, self.weight_reward))
            self.train()


if __name__ == '__main__':
    num_agent = 8
    envs = [gym.make('CartPole-v0') for _ in range(num_agent)]
    envs = [env.unwrapped for env in envs]
    test = svpg_reinforce(envs, gamma=0.99, learning_rate=1e-3, episode=100000, render=False, temperature=5.0)
    test.run()