import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import pandas as pd
from environment import Environment

# Hyper Parameters
BATCH_SIZE = 8
LR = 0.01  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # discount factor
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 80000
FILE_SIZE = 10000
Cache_size = 100  # from 100-200
env = Environment(FILE_SIZE, Cache_size)
N_STATES = 2
N_ACTIONS = 2
# files_matrix = torch.ones(10000).view(100, 100)         # files_matrix

'''
    将神经网络输入设置为当前状态，然后将状态输入神经网络
    这里输入的时候把每一个文件拆开，每一个状态(on,qon)，输入缓存大小的次数，然后到下一步
    所以这里就每一个文件都会输出两个动作值，即丢弃动作值和保存动作值。
    将整个服务器文件转换为一个矩阵存储，缓存用一个浮点数组存储，
    整数部分代表横坐标，小数部分代表纵坐标，
'''


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(10, N_ACTIONS)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


input_layer = torch.unsqueeze(torch.FloatTensor([1, 2]), 0)
net = Net()
out = net.forward(input_layer)
print(out)
print(torch.max(out, 1)[1].data.numpy())


class DQN:
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        # MEMORY_CAPACITY 行，也就是记忆库的大小，N_STATES * 2 + 2 是列，
        # 每一行能存储的列数，两个状态，动作和奖励各占一列
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.SmoothL1Loss()

    def choose_action(self, state: np.array, file_set: np.array):
        if np.random.uniform() < EPSILON:
            actions = np.array([], dtype=np.int32)
            temp_file = np.unique(np.union1d(state, file_set))
            inputs = np.array([], dtype=np.float)
            for data in temp_file:
                # 将坐标转换为小数，防止因坐标过大导致误差偏大
                x = np.int(data)
                y = np.float((data - x) * 1000 / env.cache_sqrt)
                x = np.float(x)
                inputs = np.append(inputs, x)
                inputs = np.append(inputs, y)
                inputs = torch.from_numpy(inputs)
                inputs = torch.unsqueeze(inputs, 0)
                # 这里输出的是动作值
                action_value = self.eval_net.forward(inputs)
                # 选择动作值大的那个动作，由于只有两个动作，所以将索引作为动作值
                # 这就是这里取索引的原因
                action = torch.max(action_value, 1)[1].data.numpy()
                actions = np.append(action[0])
        else:
            actions = np.random.randint(0, N_ACTIONS, Cache_size)
        return actions

    def store_transition(self, s, a, r, s_):
        # 就可以理解为  list.append()， 将s, a, r, s_, 拼接起来
        # s.size() = s_.size() = 100, a.size() = 100, r = float
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        # 这里是取 index 行，该行存储 当前的 transition
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        # 取 sample_index 这一行
        b_memory = self.memory[sample_index, :]
        bs_state = torch.FloatTensor(b_memory[:, :Cache_size])
        bs_a = torch.IntTensor(b_memory[:, :Cache_size:Cache_size * 2])
        bs_r = torch.FloatTensor(b_memory[:, :Cache_size * 2:Cache_size * 2 +
                                          1])
        bs_state_ = torch.IntTensor(b_memory[:, Cache_size * 2 + 1:])
        # 这里应该每次更新缓存大小的次数，
        for s, s_, a in zip(bs_state, bs_state_, bs_a):
            q_eval = self.eval_net(s).gather(1, a)
            # 去掉反向传播
            q_next = self.target_net(s_).detach()
            q_target = bs_r + GAMMA * q_next.max(1)[0]
            # q_target = bs_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
            loss = self.loss_func(q_eval, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


# def state_to_matrix(state):
#     files_matrix = torch.ones(10000).view(100, 100)
#     x_data = env.to_x_data(s)
#     y_data = env.to_y_data(s, x_data)
#     for x, y in enumerate(x_data, y_data):
#         files_matrix[x][y] = 0
#     return files_matrix

dqn = DQN()
print('\n Collecting experience...')
for i_episode in range(400):
    # s 是一个数组，整数部分是 x_data,小数部分是 y_data
    s = env.reset()
    ep_r = 0
    while True:
        a = dqn.choose_action(s)
        env.step()
# CUDA_VISIBLE_DEVICE=1
