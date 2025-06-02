import gym

# 환경 로드
env = gym.make("ALE/Pong-v5")

# 환경 로드 확인
print("환경이 성공적으로 로드되었습니다: ", env)
