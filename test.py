import gym
import random
import numpy as np
env = gym.make('Pendulum-v0')
success = 0
for i_episode in range(5000):
    """
    for t in range(500):
        
        x = 0
        score = 0
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        score += reward*/
        """
    x, y, score = [], [], 0
    state = env.reset()
    t = 0
    env.render()
    while True:
        action = [random.uniform(-2,2)]
        x.append(state)
        #y.append([1, 0] if action == 0 else [0, 1]) # 记录数据 
        state, reward, done, _ = env.step(action) # 执行动作
        score += reward + 16
        t += 1
        if t % 10 == 0:
            print("{}-{}-{},score is {}".format(i_episode,t + 1,success,score))
        else:
            pass
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            if t < 199:
                success += 1
                break
            else:
                break
        else:
            pass
        
env.close()