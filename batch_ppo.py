import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import transforms
from collections import deque
from common.utils import plot_total_reward
from updated_model import CNNPolicyNet, CNNValueNet
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)   

# 전처리 정의 (grayscale + resize + tensor 변환)
def preprocess(state):
    # state: np.array, shape [210, 160, 3], dtype=uint8
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)  # → [210,160]
    state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
    return torch.tensor(state, dtype=torch.float32, device=device) / 255.0


# def stack_frames(frames, new_frame, is_new_episode):
#     processed = preprocess(new_frame)
#     if is_new_episode:
#         for _ in range(frames.maxlen):
#             frames.append(processed)
#     else:
#         frames.append(processed)
#     stacked_state = torch.stack(list(frames), dim=0)  # [4, 84, 84]
#     return stacked_state.unsqueeze(0).to(device)  # [1, 4, 84, 84]

def normalize_reward(reward):
    # 방법 1: 단순 스케일링
    # return reward / 21.0
    
    # 방법 2: 더 세밀한 보상 체계
    if reward > 0:  # 점수를 얻었을 때
        return 1.0
    elif reward < 0:  # 점수를 잃었을 때
        return -1.0
    else:  # 그 외의 경우
        return 0.01  # 작은 생존 보상

class Agent:
    def __init__(self, action_size):
        self.gamma = 0.99
        self.lr_pi = 0.0001
        self.lr_v = 0.0003
        self.entropy_coef = 0.1  # 🔥 추가: 탐험 보너스 계수
        self.max_grad_norm = 0.5  # Gradient Clipping을 위한 임계값 추가

        self.pi = CNNPolicyNet(action_size).to(device)  # in_channels=4
        self.v = CNNValueNet().to(device)

        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)


    def get_action(self, state):
        self.pi.eval()  # 🔥 여기 꼭 들어가야 함
        with torch.no_grad():
            probs = self.pi(state)  # 여기서 배치 크기 1
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
        # Value Network의 gradient clipping 추가
        nn.utils.clip_grad_norm_(self.v.parameters(), self.max_grad_norm)
        self.optimizer_v.step()

        self.optimizer_pi.zero_grad()
        loss_pi.backward()
        # Policy Network의 gradient clipping 추가
        nn.utils.clip_grad_norm_(self.pi.parameters(), self.max_grad_norm)
        self.optimizer_pi.step()

def update_frame_stack(frame_stack, new_frame, is_new_episode):
    if is_new_episode:
        frame_stack[:] = new_frame.repeat(4, 1, 1)
    else:
        frame_stack[:-1] = frame_stack[1:].clone()  # ✅ clone()으로 memory overlap 방지
        frame_stack[-1] = new_frame
    return frame_stack.unsqueeze(0)


# 환경 및 학습 설정
episodes = 100000
eps_clip = 0.2
K_epochs = 4

env = gym.make("ALE/Pong-v5")
action_size = env.action_space.n
agent = Agent(action_size=action_size)
reward_history = []

multi_rollout = []
multi_episode_count = 10

# 4개 프레임을 모아서 하나의 상태로 만들기
frame_stack = torch.zeros((4, 84, 84), dtype=torch.float32, device=device)

for episode in range(episodes):
    state = env.reset()[0]
    new_frame = preprocess(state)

    frame_stack = torch.zeros((4, 84, 84), dtype=torch.float32, device=device)  # 에피소드마다 초기화
    stacked_state = update_frame_stack(frame_stack, new_frame, is_new_episode=True)

    done = False
    total_reward = 0
    
    # rollout 관련 (ppo 알고리즘 관련)
    rollout = []

    while not done:
        action, log_prob = agent.get_action(stacked_state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        new_frame = preprocess(next_state) 
        next_stacked_state = update_frame_stack(frame_stack, new_frame, is_new_episode=False)

        # 보상 정규화 추가
        normalized_reward = normalize_reward(reward)

        done = terminated or truncated

        # ✅ rollout에 프레임스택 상태와 함께 저장
        rollout.append((stacked_state, action, log_prob, normalized_reward, next_stacked_state, done))

        stacked_state = next_stacked_state
        total_reward += reward

    multi_rollout.extend(rollout)
    reward_history.append(total_reward)

    if episode % 10 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward:.1f}")
    
    if episode % 5000 == 0:
        torch.save(agent.pi.state_dict(), f"./batch_models/ppo_pi_ep{episode}.pt")
        print(f"🧠 Saved model at episode {episode}")
    
    # 다중 에폭 학습 (Multiple Epochs)
    if (episode + 1) % multi_episode_count == 0:
        agent.pi.train()  
        for _ in range(K_epochs):
            agent.update(multi_rollout, eps_clip)
        multi_rollout = []

plot_total_reward(reward_history)
