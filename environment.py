# from gym import spaces
import numpy as np
from collections import Counter
# from gym.utils import seeding


class Environment:
    '''
        动作就是 n_actions，这个也是缓存大小
        状态 S=<B,L>
        L=user_requests
        B是一个二元组的集合，每一个二元组是(内容块i, 一个归一化增量)
    '''
    '''
        先写一个状态简单的，那就将系统状态只设置为基站缓存
        用户状态先不设置
    '''
    def __init__(self, files: int, cache_size: int):
        '''
            为了防止存坐标时顺序出错，这里用一个浮点数表示横纵坐标
            整数部分表示横坐标，小数部分表示纵坐标
            如   12.34,   表示 (12, 34)这一个文件
            self.server_file = np.ones((100, 100), dtype=int)
            S = <B,L> = <on, qon, L>
            L = <l1, l2, lK>, lk 是user k 请求的文件集
            qo+1 = qo + delta_o
            delta_o = co / sum(co)
            co 表示文件 o 被请求的总次数
            设缓存大小的平方 = 文件总数
        '''
        self.server_file = files  # server file numbers
        self.server_sqrt = np.sqrt(files).astype(int)
        self.cache_size = cache_size
        self.request_nums = 0
        self.cache_sqrt = np.sqrt(cache_size).astype(int)
        # qon and con matrix
        self.qon = np.zeros((self.server_sqrt, self.server_sqrt))
        self.threshold = np.float64(0)
        self.con = np.zeros((self.server_sqrt, self.server_sqrt))
        self.bs_state = None  # BS state
        # self.user_request = None        # user state
        self.state = None  # system state = (BSstate, userstate)
        self.pre_state = None
        self.eta = 0.7  # 削减奖励
        # self.state = (self.bs_state, self.user_request)

    def to_position(self, xdata, ydata):
        return xdata + (ydata / 1000)

    def to_x_data(self, arr):
        return arr.astype(int)

    def to_y_data(self, arr, x_data):
        return ((arr - x_data) * 1000).astype(int)

    def reset(self) -> np.array:
        # replace=False 表示生成不同的随机数，默认为 True
        # 保证每次初始随机初始化一样
        np.random.seed(1)
        # sqrt_state = self.cache_size
        position = np.random.choice(np.arange(self.cache_size),
                                    self.cache_sqrt + self.cache_sqrt,
                                    replace=False)
        x_data, y_data = position[:self.cache_sqrt], position[self.cache_size:]
        for x in x_data:
            for y in y_data:
                self.con[x][y] += 1
        # self.user_request = None
        self.bs_state = self.to_position(x_data, y_data)
        # self.state = (self.bs_state, self.user_request)
        self.state = self.bs_state
        self.pre_state = self.state
        return self.state

    def step(self, actions: np.array, file_set: np.array,
             request_file: np.array):
        '''
            file_set 是与 actions 一一对应的
            所以不能取request_file的集合。
            actions 表示采取动作的集合，
            new_files 偶数位表示横坐标，奇数位表示纵坐标
            新缓存的集合是 D+，也就是 new_files
            pre_state and state 的交集是 D*
            # file_set 是当前缓存 + 新取的文件集
        '''
        temp_bs_state = np.array([])
        temp_evict = np.array([])
        # 归一化的基数
        self.request_nums += request_file.size
        for action, data in zip(actions, file_set):
            if action:
                temp_bs_state = np.append(temp_bs_state, data)
            else:
                temp_evict = np.append(temp_evict, data)
        temp_qon = self.qon
        # np.unravel_index(np.argmax(arr), arr.shape) 求最大值的索引
        while temp_bs_state.size < self.cache_size:
            pos = np.unravel_index(np.argmax(temp_qon, temp_qon.shape))
            temp_bs_state = np.append(temp_bs_state, int(pos[0]) + int(pos[1])/1000)
            temp_qon[pos[0]][pos[1]] = 0
        if temp_bs_state.size >= self.cache_size:
            # D*
            both_cache = np.intersect1d(self.state, self.pre_state)
            request_count = Counter(request_file)
            # D*
            both_cache = np.intersect1d(self.state, self.pre_state)
            cache_hit = np.intersect1d(self.state, temp_bs_state)
            # 新缓存的文件集 D+
            newly_cache = np.setdiff1d(cache_hit, temp_bs_state)
            # D-
            pre_evict = np.setdiff1d(both_cache, self.pre_state)
            # D+ U D-
            nega = np.union1d(pre_evict, request_file)
            vco_both, vco_new, mco = 0, 0, 0
            for data, counts in request_count.items():
                x = self.to_x_data(data)
                y = self.to_y_data(data, x)
                self.con[x][y] += counts
                if data in self.state:
                    # qon = qon + delta
                    self.qon[x][y] += counts / request_file.size
                if data in both_cache:
                    vco_both += counts
                if data in newly_cache:
                    vco_new += counts
                if data in nega:
                    mco += counts
            positive = (vco_both + self.eta * vco_new) / self.request_nums
            self.pre_state = self.state
            self.state = temp_bs_state[:self.cache_size]
            negative = mco / self.request_nums
            reward = positive - negative
            return (self.state, reward)
            # D-
            # pre_evict = np.setdiff1d(self.pre_state, both_cache)
            # negative_file = np.union1d(pre_evict, newly_cache)
            # negative =
        # return (self.state, reward)
