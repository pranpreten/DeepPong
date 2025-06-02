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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
        self.entropy_coef = 0.01  # 🔥 추가: 탐험 보너스 계수

        self.pi = CNNPolicyNet(action_size).to(device)  # in_channels=4
        self.v = CNNValueNet().to(device)

        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state):
        probs = self.pi(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, rollout, eps_clip):
        states, actions, log_probs, rewards, next_states, dones = zip(*rollout)

        states = torch.cat(states).to(device)
        next_states = torch.cat(next_states).to(device)
        log_probs = torch.stack(log_probs).unsqueeze(1).detach()
        actions = torch.tensor(actions).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        # ─────────────────────────────────────
        # 🎯 1. Target, Value, Advantage
        with torch.no_grad():
            target = rewards + self.gamma * self.v(next_states) * (1 - dones)

        values = self.v(states)
        advantages = (target - values).detach()

        # advantages 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 🎯 2. Value Loss
        loss_v = F.mse_loss(values, target)

        # 🎯 3. PPO Policy Loss (정책 변화 제한)
        probs = self.pi(states)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions.squeeze(1)).unsqueeze(1)
        
        entropy = dist.entropy().mean()

        ratio = (new_log_probs - log_probs).exp()
        clipped_ratio = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip)
        loss_pi = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # 🔥 탐험 유도 항 추가 (엔트로피가 클수록 보너스)
        loss_pi -= self.entropy_coef * entropy
        # ─────────────────────────────────────

        # 🎯 4. Optimization
        self.optimizer_v.zero_grad()
        loss_v.backward()
        self.optimizer_v.step()

        self.optimizer_pi.zero_grad()
        loss_pi.backward()
        self.optimizer_pi.step()




# 환경 및 학습 설정
episodes = 50000
eps_clip = 0.2
K_epochs = 4

env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
action_size = env.action_space.n
agent = Agent(action_size=action_size)
reward_history = []

# 4개 프레임을 모아서 하나의 상태로 만들기
frame_stack = deque(maxlen=4)

for episode in range(episodes):
    state = env.reset()[0]
    stacked_state = stack_frames(frame_stack, state, is_new_episode=True)

    done = False
    total_reward = 0
    
    # rollout 관련 (ppo 알고리즘 관련)
    rollout = []

    while not done:
        action, log_prob = agent.get_action(stacked_state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_stacked_state = stack_frames(frame_stack, next_state, is_new_episode=False)

        done = terminated or truncated

        # ✅ rollout에 프레임스택 상태와 함께 저장
        rollout.append((stacked_state, action, log_prob, reward, next_stacked_state, done))

        # agent.update(stacked_state, log_prob, reward, next_stacked_state, done)

        stacked_state = next_stacked_state
        total_reward += reward

    reward_history.append(total_reward)

    if episode % 10 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward:.1f}")
    
    if episode % 1000 == 0:
        torch.save(agent.pi.state_dict(), f"./advanced_ppo_models/ppo_pi_ep{episode}.pt")
        print(f"🧠 Saved model at episode {episode}")
    
    # ppo 는 on-policy 방식이기 때문에 가장 최근 policy로만 업데이트해야 함
    # 다중 에폭 학습 (Multiple Epochs)
    for _ in range(K_epochs):
        agent.update(rollout, eps_clip)

plot_total_reward(reward_history)
