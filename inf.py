import gym
import torch
import numpy as np
from gym.wrappers import AtariPreprocessing, FrameStack
from updated_model import CNNPolicyNet  # 학습에 사용했던 동일한 모델 클래스

# 하이퍼파라미터
ENV_NAME = "PongNoFrameskip-v4"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 환경 설정
env = gym.make(ENV_NAME, render_mode="human")
env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False)
env = FrameStack(env, 4)

# 모델 불러오기
policy_net = CNNPolicyNet(action_size=env.action_space.n).to(DEVICE)
# policy_net.load_state_dict(torch.load("ppo_policy.pth"))
policy_net.load_state_dict(torch.load("ppo_policy.pth", map_location=DEVICE))
policy_net.eval()

# 환경 실행
obs = env.reset()
done = False

while not done:
    obs_tensor = torch.from_numpy(np.array(obs)).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        logits = policy_net(obs_tensor)
        action = torch.argmax(logits, dim=1).item()
    obs, reward, done, info = env.step(action)

env.close()
