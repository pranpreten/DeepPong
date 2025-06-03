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
from gym.vector import AsyncVectorEnv


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
        # state: [batch, 4, 84, 84]
        probs = self.pi(state)  # shape: [batch, action_size]
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()  # shape: [batch]
        log_probs = dist.log_prob(actions)  # shape: [batch]
        return actions, log_probs

    def update(self, rollout):
        states, actions, log_probs, rewards, next_states, dones = zip(*rollout)

        states = torch.cat(states)
        next_states = torch.cat(next_states)
        log_probs = torch.stack(log_probs).unsqueeze(1).detach()
        actions = torch.tensor(actions).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            print("sdafasfas", next_states.shape)
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

def make_env():
    def _init():
        return gym.make("ALE/Pong-v5", render_mode="rgb_array")
    return _init

if __name__ == "__main__":
    # 환경 및 학습 설정
    episodes = 10000
    num_envs = 4  # 병렬로 돌릴 환경 개수
    env = AsyncVectorEnv([make_env() for _ in range(num_envs)])
    # action_size = env.action_space.n
    agent = Agent(action_size=6)

    reward_history = []

    frame_stacks = [deque(maxlen=4) for _ in range(num_envs)]

    for episode in range(episodes):
        obs = env.reset()[0]  # shape: [4, 210, 160, 3]
        stacked_states = []
        for i in range(num_envs):
            stacked = stack_frames(frame_stacks[i], obs[i], is_new_episode=True)
            stacked_states.append(stacked)

        stacked_states = torch.cat(stacked_states)

        done_flags = [False] * num_envs
        total_rewards = [0.0] * num_envs
        rollouts = [[] for _ in range(num_envs)]

        while not all(done_flags):
            actions, log_probs = agent.get_action(stacked_states)

            actions_np = actions.cpu().numpy()
            next_obs, rewards, terminateds, truncateds, _ = env.step(actions_np)
            dones = np.logical_or(terminateds, truncateds)

            next_stacked_states = []
            for i in range(num_envs):
                next_stacked = stack_frames(frame_stacks[i], next_obs[i], is_new_episode=dones[i])
                next_stacked_states.append(next_stacked)

                if not done_flags[i]:
                    rollouts[i].append((
                        stacked_states[i],      # 현재 상태 [1, 4, 84, 84]
                        actions[i].item(),                   # 액션 정수값
                        log_probs[i],                        # log π(a|s)
                        rewards[i],                          # 보상
                        next_stacked,           # 다음 상태 [1, 4, 84, 84]
                        dones[i]                             # 종료 여부
                    ))
                    total_rewards[i] += rewards[i]

                    if dones[i]:
                        done_flags[i] = True

            stacked_states = torch.cat(next_stacked_states)  # 다음 step을 위한 입력 갱신

        # ▶ 모든 환경 rollout을 하나로 합치기
        flat_rollout = [transition for rollout in rollouts for transition in rollout]

        # ▶ 에이전트 업데이트
        agent.update(flat_rollout)

        # ▶ 평균 리워드 기록
        avg_reward = np.mean(total_rewards)
        reward_history.append(avg_reward)

        if episode % 10 == 0:
            print(f"[Episode {episode}] Average Reward: {avg_reward:.1f}")
            



    # plot_total_reward(reward_history)
