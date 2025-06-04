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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)   

# 전처리 정의 (grayscale + resize + tensor 변환)
def preprocess(state):
    # state: np.array, shape [210, 160, 3], dtype=uint8
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)  # → [210,160]
    state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
    return torch.tensor(state, dtype=torch.float32, device=device) / 255.0


def normalize_reward(reward):
    if reward > 0:  # 점수를 얻었을 때
        return 1.0
    elif reward < 0:  # 점수를 잃었을 때
        return -1.0
    else:  # 그 외의 경우
        return 0.0  # 생존 보상 제거 (더 적극적인 공격 유도)

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
        
        # 학습률 스케줄러 추가
        self.scheduler_pi = optim.lr_scheduler.StepLR(self.optimizer_pi, step_size=100000//4, gamma=0.5)
        self.scheduler_v = optim.lr_scheduler.StepLR(self.optimizer_v, step_size=100000//4, gamma=0.5)

    def get_action(self, state):
        self.pi.eval()  # 🔥 여기 꼭 들어가야 함
        with torch.no_grad():
            probs = self.pi(state)  # 여기서 배치 크기 1
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


    def update(self, rollout, eps_clip, batch_size=512):
        states, actions, log_probs, rewards, next_states, dones = zip(*rollout)

        states = torch.cat(states).to(device)
        next_states = torch.cat(next_states).to(device)
        log_probs = torch.stack(log_probs).unsqueeze(1).detach()
        actions = torch.tensor(actions).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            target = rewards + self.gamma * self.v(next_states) * (1 - dones)

        values = self.v(states)
        advantages = (target - values).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 학습은 mini-batch로 나눠서 처리
        data_size = states.size(0)
        for i in range(0, data_size - batch_size + 1, batch_size):
            s = states[i:i+batch_size]
            a = actions[i:i+batch_size]
            logp_old = log_probs[i:i+batch_size]
            adv = advantages[i:i+batch_size]
            tgt = target[i:i+batch_size]

            # Value Loss
            v_pred = self.v(s)
            loss_v = F.mse_loss(v_pred, tgt)

            # Policy Loss
            probs = self.pi(s)
            dist = torch.distributions.Categorical(probs)
            logp_new = dist.log_prob(a.squeeze(1)).unsqueeze(1)
            entropy = dist.entropy().mean()

            ratio = (logp_new - logp_old).exp()
            clipped_ratio = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip)
            loss_pi = -torch.min(ratio * adv, clipped_ratio * adv).mean()
            loss_pi -= self.entropy_coef * entropy

            # Gradient update
            self.optimizer_v.zero_grad()
            loss_v.backward()
            nn.utils.clip_grad_norm_(self.v.parameters(), self.max_grad_norm)
            self.optimizer_v.step()

            self.optimizer_pi.zero_grad()
            loss_pi.backward()
            nn.utils.clip_grad_norm_(self.pi.parameters(), self.max_grad_norm)
            self.optimizer_pi.step()

        # 스케줄러 업데이트는 1번만
        self.scheduler_pi.step()
        self.scheduler_v.step()


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

env = gym.make("ALE/Pong-v5", frameskip=4)
action_size = env.action_space.n
agent = Agent(action_size=action_size)
reward_history = []

multi_rollout = []
torch.cuda.empty_cache()
multi_episode_count = 5

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
        rollout.append((
            stacked_state.detach(), 
            action, 
            log_prob.detach(), 
            normalized_reward, 
            next_stacked_state.detach(), 
            done
        ))

        stacked_state = next_stacked_state
        total_reward += reward

    multi_rollout.extend(rollout)
    reward_history.append(total_reward)

    if episode % 10 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward:.1f}")
    
    if episode % 1000 == 0:
        torch.save(agent.pi.state_dict(), f"./ppo_test_1_models/ppo_pi_ep{episode}.pt")
        print(f"🧠 Saved model at episode {episode}")
    
    # 다중 에폭 학습 (Multiple Epochs)
    if (episode + 1) % multi_episode_count == 0:
        agent.pi.train()  
        for _ in range(K_epochs):
            agent.update(multi_rollout, eps_clip)
        multi_rollout = []

plot_total_reward(reward_history)
