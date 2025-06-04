import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import transforms
from collections import deque
from common.utils import plot_total_reward
from model import CNNPolicyNet, CNNValueNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 전처리 정의 (grayscale + resize + tensor 변환)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((84, 84)),
    transforms.ToTensor()
])

def preprocess(state):
    state = transform(state).squeeze(0)  # [84, 84]
    return state

def stack_frames(frames, new_frame, is_new_episode):
    processed = preprocess(new_frame)
    if is_new_episode:
        for _ in range(frames.maxlen):
            frames.append(processed)
    else:
        frames.append(processed)
    stacked_state = torch.stack(list(frames), dim=0)  # [4, 84, 84]
    return stacked_state.unsqueeze(0).to(device)  # [1, 4, 84, 84]

class Agent:
    def __init__(self, action_size):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005

        self.pi = CNNPolicyNet(action_size).to(device)  # in_channels=4
        self.v = CNNValueNet().to(device)

        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state):
        probs = self.pi(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, state, log_prob, reward, next_state, done):
        reward = torch.tensor([reward], dtype=torch.float32).to(device)
        done = torch.tensor([done], dtype=torch.float32).to(device)

        with torch.no_grad():
            target = reward + self.gamma * self.v(next_state) * (1 - done)

        value = self.v(state)
        loss_v = F.mse_loss(value, target)

        self.optimizer_v.zero_grad()
        loss_v.backward()
        self.optimizer_v.step()

        advantage = (target - value).detach()
        loss_pi = -log_prob * advantage

        self.optimizer_pi.zero_grad()
        loss_pi.backward()
        self.optimizer_pi.step()

# 환경 및 학습 설정
episodes = 1000
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
action_size = env.action_space.n
agent = Agent(action_size=action_size)
reward_history = []

frame_stack = deque(maxlen=4)

for episode in range(episodes):
    state = env.reset()[0]
    stacked_state = stack_frames(frame_stack, state, is_new_episode=True)

    done = False
    total_reward = 0

    while not done:
        action, log_prob = agent.get_action(stacked_state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_stacked_state = stack_frames(frame_stack, next_state, is_new_episode=False)

        done = terminated or truncated
        agent.update(stacked_state, log_prob, reward, next_stacked_state, done)

        stacked_state = next_stacked_state
        total_reward += reward

    reward_history.append(total_reward)

    if episode % 10 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward:.1f}")

plot_total_reward(reward_history)
