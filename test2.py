import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import transforms
from collections import deque
from common.utils import plot_total_reward
from test1_model import CNNPolicyNet, CNNValueNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

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
        # 클리핑추가
        self.eps_clip = 0.1

        self.pi = CNNPolicyNet(action_size).to(device)  # in_channels=4
        self.v = CNNValueNet().to(device)

        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state):
        probs = self.pi(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, rollout):
        states, actions, log_probs, rewards, next_states, dones = zip(*rollout)

        states = torch.cat(states)
        next_states = torch.cat(next_states)
        log_probs = torch.stack(log_probs).unsqueeze(1).detach()
        actions = torch.tensor(actions).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            target = rewards + self.gamma * self.v(next_states) * (1 - dones)

        values = self.v(states)
        advantages = (target - values).detach()

        loss_v = F.mse_loss(values, target)

        self.optimizer_v.zero_grad()
        loss_v.backward()
        self.optimizer_v.step()

        probs = self.pi(states)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions.squeeze()).unsqueeze(1)

        ratio = (new_log_probs - log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        loss_pi = -torch.min(surr1, surr2).mean()

        self.optimizer_pi.zero_grad()
        loss_pi.backward()
        self.optimizer_pi.step()


# 환경 및 학습 설정
episodes = 10000
env = gym.make("ALE/Pong-v5")
action_size = env.action_space.n
agent = Agent(action_size=action_size)
reward_history = []

frame_stack = deque(maxlen=4)

for episode in range(episodes):
    state = env.reset()[0]
    stacked_state = stack_frames(frame_stack, state, is_new_episode=True)

    done = False
    total_reward = 0

    rollout = []

    while not done:
        action, log_prob = agent.get_action(stacked_state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_stacked_state = stack_frames(frame_stack, next_state, is_new_episode=False)

        done = terminated or truncated
        rollout.append((stacked_state, action, log_prob, reward, next_stacked_state, done))

        stacked_state = next_stacked_state
        total_reward += reward

    reward_history.append(total_reward)

    agent.update(rollout)

    if episode % 10 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward:.1f}")

plot_total_reward(reward_history)
