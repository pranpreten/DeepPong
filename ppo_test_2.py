import os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from gym.vector import AsyncVectorEnv
from gym.wrappers import AtariPreprocessing, FrameStack
from tqdm import trange
import matplotlib.pyplot as plt
from updated_model import CNNPolicyNet, CNNValueNet
from tqdm import tqdm
import cv2
# ===== 하이퍼파라미터 =====
ENV_NAME = "PongNoFrameskip-v4"
NUM_ENVS = 64
ROLLOUT_STEPS = 128
TOTAL_TIMESTEPS = 5000000
# TOTAL_TIMESTEPS = 100000
PPO_EPOCHS = 4
MINI_BATCH_SIZE = 1024
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.1
LR = 2.5e-4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# ==== 환경 생성 ====
def make_env():
    def thunk():
        env = gym.make(ENV_NAME)
        env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False)
        env = FrameStack(env, 4)
        return env
    return thunk

envs = AsyncVectorEnv([make_env() for _ in range(NUM_ENVS)])

policy_net = CNNPolicyNet(envs.single_action_space.n).to(DEVICE)
value_net = CNNValueNet().to(DEVICE)

optimizer_pi = optim.Adam(policy_net.parameters(), lr=LR, eps=1e-5)
optimizer_v = optim.Adam(value_net.parameters(), lr=LR, eps=1e-5)

# ==== 학습 루프 ====
obs = envs.reset()
obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

reward_history = []
episode_rewards = np.zeros(NUM_ENVS)
max_reward_achieved = -float("inf")

num_updates = TOTAL_TIMESTEPS // (NUM_ENVS * ROLLOUT_STEPS)

with tqdm(total=TOTAL_TIMESTEPS, desc="PPO Training") as pbar:
    for update in trange(num_updates, desc="Training"):
        storage = {'obs': [], 'actions': [], 'logprobs': [], 'rewards': [], 'values': [], 'dones': []}

        for step in range(ROLLOUT_STEPS):
            with torch.no_grad():
                probs = policy_net(obs)
                dist = Categorical(probs)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)
                values = value_net(obs).squeeze()

            next_obs, reward, done, info = envs.step(actions.cpu().numpy())

            storage['obs'].append(obs.cpu().numpy())
            storage['actions'].append(actions.cpu().numpy())
            storage['logprobs'].append(logprobs.cpu().numpy())
            storage['values'].append(values.cpu().numpy())
            storage['rewards'].append(reward)
            storage['dones'].append(done)

            obs = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE)
            episode_rewards += reward

            for i, d in enumerate(done):
                if d:
                    reward_history.append(episode_rewards[i])
                    episode_rewards[i] = 0

        # → numpy → torch 변환
        for k in storage:
            if k in ['rewards', 'values']:  # float 데이터
                storage[k] = torch.tensor(np.asarray(storage[k]), dtype=torch.float32, device=DEVICE)
            elif k in ['obs']:  # observation (이미 float)
                storage[k] = torch.tensor(np.asarray(storage[k]), dtype=torch.float32, device=DEVICE)
            else:  # actions, logprobs, dones (정수 or bool)
                storage[k] = torch.tensor(np.asarray(storage[k]), device=DEVICE)

        # Advantage 계산 (GAE)
        with torch.no_grad():
            next_value = value_net(obs).squeeze()

        advantages = torch.zeros_like(storage['rewards'], device=DEVICE)
        last_adv = 0
        for t in reversed(range(ROLLOUT_STEPS)):
            mask = 1.0 - storage['dones'][t].float()
            delta = storage['rewards'][t] + GAMMA * next_value * mask - storage['values'][t]
            advantages[t] = last_adv = delta + GAMMA * GAE_LAMBDA * mask * last_adv
            next_value = storage['values'][t]

        returns = advantages + storage['values']

        # Flatten batch
        obs_batch = storage['obs'].reshape(-1, 4, 84, 84)
        actions_batch = storage['actions'].reshape(-1)
        logprobs_batch = storage['logprobs'].reshape(-1)
        values_batch = storage['values'].reshape(-1)
        returns_batch = returns.reshape(-1)
        advantages_batch = advantages.reshape(-1)
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

        inds = np.arange(obs_batch.shape[0])
        for _ in range(PPO_EPOCHS):
            np.random.shuffle(inds)
            for start in range(0, len(inds), MINI_BATCH_SIZE):
                mb_inds = inds[start:start+MINI_BATCH_SIZE]

                logits = policy_net(obs_batch[mb_inds])
                dist = Categorical(logits)
                new_logprobs = dist.log_prob(actions_batch[mb_inds])
                entropy = dist.entropy().mean()

                ratio = (new_logprobs - logprobs_batch[mb_inds]).exp()
                surr1 = ratio * advantages_batch[mb_inds]
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages_batch[mb_inds]
                policy_loss = -torch.min(surr1, surr2).mean()

                values_pred = value_net(obs_batch[mb_inds]).squeeze()
                value_loss = F.mse_loss(values_pred, returns_batch[mb_inds])

                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

                optimizer_pi.zero_grad()
                optimizer_v.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(policy_net.parameters()) + list(value_net.parameters()), MAX_GRAD_NORM)
                optimizer_pi.step()
                optimizer_v.step()

        if update % 10 == 0:
            torch.save(policy_net.state_dict(), "ppo_policy.pth")
            print(f"Saved at update {update} | Recent reward: {np.mean(reward_history[-10:])}")

# ==== 보상 그래프 ====
plt.plot(reward_history)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("PPO Pong Training")
plt.savefig("atari_pong_reward.png")
plt.close()

from gym.wrappers import AtariPreprocessing, FrameStack

policy_net.load_state_dict(torch.load("ppo_policy.pth"))
policy_net.eval()

env = gym.make(ENV_NAME, render_mode="rgb_array")
env = AtariPreprocessing(env, frame_skip=4, scale_obs=False)
env = FrameStack(env, 4)

obs = env.reset()
done = False
frames = []

while not done:
    obs_tensor = torch.from_numpy(np.array(obs)).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        logits = policy_net(obs_tensor)
        action = torch.argmax(logits, dim=1).item()
    obs, reward, done, info = env.step(action)

    frame = env.render()
    frames.append(frame)

env.close()

video_name = "atari_pong_play.mp4"
os.makedirs("video", exist_ok=True)
video_path = os.path.join("video", video_name)

height, width, _ = frames[0].shape
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
out.release()
print(f"Video saved as {video_name}")