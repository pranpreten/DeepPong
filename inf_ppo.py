import gym
import torch
import numpy as np
from collections import deque
from torchvision import transforms
from model import CNNPolicyNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 전처리 정의
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((84, 84)),
    transforms.ToTensor()
])

def preprocess(state):
    state = transform(state).squeeze(0)
    return state

def stack_frames(frames, new_frame, is_new_episode):
    processed = preprocess(new_frame)
    if is_new_episode:
        for _ in range(frames.maxlen):
            frames.append(processed)
    else:
        frames.append(processed)
    stacked_state = torch.stack(list(frames), dim=0)
    return stacked_state.unsqueeze(0).to(device)

# 모델 로딩
action_size = 6  # Pong은 6개 액션
pi = CNNPolicyNet(action_size).to(device)
pi.load_state_dict(torch.load("./ppo_models/ppo_pi_ep1000.pt", map_location=device))
pi.eval()

# 환경 설정
env = gym.make("ALE/Pong-v5", render_mode="human")
frame_stack = deque(maxlen=4)

state = env.reset()[0]
stacked_state = stack_frames(frame_stack, state, is_new_episode=True)

done = False
total_reward = 0

while not done:
    with torch.no_grad():
        probs = pi(stacked_state)
        action = torch.argmax(probs, dim=1).item()

    next_state, reward, terminated, truncated, _ = env.step(action)
    stacked_state = stack_frames(frame_stack, next_state, is_new_episode=False)
    done = terminated or truncated
    total_reward += reward

print("🏓 Game finished. Total Reward:", total_reward)
