import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from common.utils import plot_total_reward  # 기존 유틸 사용

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNet(nn.Module):  # Actor
    def __init__(self, input_dim, action_size):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x


class ValueNet(nn.Module):  # Critic
    def __init__(self, input_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Agent:
    def __init__(self, input_dim, action_size):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005

        self.pi = PolicyNet(input_dim, action_size).to(device)
        self.v = ValueNet(input_dim).to(device)

        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = self.pi(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, state, log_prob, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        reward = torch.tensor([reward], dtype=torch.float32).to(device)
        done = torch.tensor([done], dtype=torch.float32).to(device)

        # Critic 업데이트
        target = reward + self.gamma * self.v(next_state) * (1 - done)
        value = self.v(state)
        loss_v = F.mse_loss(value, target.detach())

        self.optimizer_v.zero_grad()
        loss_v.backward()
        self.optimizer_v.step()

        # Actor 업데이트
        advantage = (target - value).detach()
        loss_pi = -log_prob * advantage

        self.optimizer_pi.zero_grad()
        loss_pi.backward()
        self.optimizer_pi.step()


# 학습 파라미터
episodes = 1000
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
obs_shape = env.observation_space.shape
action_size = env.action_space.n


# observation은 이미지이기 때문에 CNN 필요 (지금은 예시용으로 flatten 사용)
input_dim = np.prod(obs_shape)

agent = Agent(input_dim=input_dim, action_size=action_size)
reward_history = []

for episode in range(episodes):
    state = env.reset()[0]
    state = state.flatten()  # 이미지 flatten (임시처리)

    done = False
    total_reward = 0

    while not done:
        action, log_prob = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = next_state.flatten()

        done = terminated or truncated
        agent.update(state, log_prob, reward, next_state, done)

        state = next_state
        total_reward += reward

    reward_history.append(total_reward)

    if episode % 100 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward:.1f}")

plot_total_reward(reward_history)
