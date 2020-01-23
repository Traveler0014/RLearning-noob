import random
import gym
import numpy as np
from tensorflow.keras import models, layers

env = gym.make("Pendulum-v0")  # 加载游戏环境

model = models.load_model('Pendulum-v0-nn.h5') 
model.summary()  # 打印神经网络信息

# train.py
def generate_data_one_episode():
    '''生成单次游戏的训练数据'''
    x, y, score = [], [], 0
    state = env.reset()
    while True:
        #action = [np.argmax(model.predict(np.array([state]))[0])] 
        action = [random.uniform(-2,2)]
        x.append(state)
        y.append(action) # 记录数据
        state, reward, done, _ = env.step(action) # 执行动作
        score += reward + 16
        if done:
            break
    return x, y, score

def generate_training_data(expected_score=2500):
    '''# 生成N次游戏的训练数据，并进行筛选，选择 > 100 的数据作为训练集'''
    data_X, data_Y, scores = [], [], []
    for i in range(2000):
        x, y, score = generate_data_one_episode()
        if score > expected_score:
            data_X += x
            data_Y += y
            scores.append(score)
        print("round:{},score:{}".format(i,score))
    print('dataset size: {}, max score: {}'.format(len(data_X), max(scores)))
    return np.array(data_X), np.array(data_Y)

# train.py
data_X, data_Y = generate_training_data()
model.compile(loss='mse', optimizer='adam')
model.fit(data_X, data_Y,epochs=500)
model.save('Pendulum-v0-nn.h5')  # 保存模型