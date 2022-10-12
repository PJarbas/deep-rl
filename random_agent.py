import gym

env = gym.make("CartPole-v0")

observation = env.reset()

while True:
    env.render()
    print(observation)
    
    action = env.action_space.sample()
    
    observation, reward, done, info = env.step(action)
    
    if done:
        break

env.close()