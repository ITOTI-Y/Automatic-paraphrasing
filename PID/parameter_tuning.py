import numpy as np
import control as ctl
from control.matlab import lsim
import pandas as pd

import matplotlib.pyplot as plt

class SystemModel:
    def __init__(self, num1, den1, num2, den2, delay):
        self.sys1 = ctl.TransferFunction(num1, den1)  # 创建传递函数 sys1
        self.sys2 = ctl.TransferFunction(num2, den2)  # 创建传递函数 sys2
        self.sys = ctl.series(self.sys1, self.sys2)  # 将 sys1 和 sys2 连接成串联系统
        self.sys = ctl.tf2ss(self.sys)  # 转换为状态空间模型
        self.delay = delay  # 设置延迟时间

    def simulate(self, t, u):
        T, y_out, _ = lsim(self.sys, u, t)  # 使用 lsim 函数进行系统仿真
        if len(t) > 1:  # 对输出信号进行延迟处理
            delay_samples = int(self.delay / (t[1] - t[0]))  # 计算延迟采样数
            y_out = np.pad(y_out, (delay_samples, 0), 'constant')
        return y_out.flatten()[t[-1]]
    
    def __repr__(self):
        return f"Initial state of the state-space model: {self.sys.A}"

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp  # 设置比例系数
        self.ki = ki  # 设置积分系数
        self.kd = kd  # 设置微分系数

    def compute_control(self, t, e):
        if len(e) == 0 or len(t) == 0:
            return 0
        elif len(e) < 2 or len(t) < 2:
            return self.kp * e[-1]
        else:
            u = self.kp * e[-1] + self.ki * np.trapz(e, t) + self.kd * np.gradient(e, t)[-1]  # 计算控制信号
            return u

class Optimizer:
    def __init__(self, num_ants, num_iterations, alpha, beta, rho, q, param_ranges):
        self.num_ants = num_ants  # 设置蚂蚁数量
        self.num_iterations = num_iterations  # 设置迭代次数
        self.alpha = alpha  # 设置 alpha 参数
        self.beta = beta  # 设置 beta 参数
        self.rho = rho  # 设置 rho 参数
        self.q = q  # 设置 q 参数
        self.param_ranges = param_ranges  # 设置参数范围
        self.pheromone = np.ones((len(param_ranges), num_ants))  # 初始化信息素矩阵

    def optimize(self, cost_func):
        best_cost = np.inf  # 初始化最佳代价
        best_solution = None  # 初始化最佳解

        for iteration in range(self.num_iterations):
            solutions = np.zeros((self.num_ants, len(self.param_ranges)))  # 初始化解向量
            costs = np.zeros(self.num_ants)  # 初始化代价向量

            for ant in range(self.num_ants):
                for i in range(len(self.param_ranges)):
                    probabilities = self.pheromone[i] ** self.alpha / ((np.sum(self.pheromone[i]) - self.pheromone[i]) ** self.beta)  # 计算概率
                    probabilities /= np.sum(probabilities)  # 归一化概率
                    solutions[ant, i] = np.random.choice(self.param_ranges[i], p=probabilities)  # 根据概率选择参数值

                costs[ant] = cost_func(*solutions[ant])  # 计算代价

                if costs[ant] < best_cost:
                    best_cost = costs[ant]  # 更新最佳代价
                    best_solution = solutions[ant]  # 更新最佳解

            self.pheromone *= (1 - self.rho)  # 更新信息素
            for ant in range(self.num_ants):
                self.pheromone[:, ant] += self.q / costs[ant]  # 更新信息素

        return best_solution, best_cost

class Simulation:
    def __init__(self, sys, w, t):
        self.sys = sys  # 设置系统模型
        self.w = w  # 设置期望输出
        self.t = t  # 设置时间向量

    def cost_func(self, kp, ki, kd):
        pid = PIDController(kp, ki, kd)  # 创建 PID 控制器
        y = np.zeros_like(self.t)  # 初始化输出向量
        e = np.zeros_like(self.t)  # 初始化误差向量
        u = np.zeros_like(self.t)  # 初始化控制向量

        for i in range(len(self.t)):
            if i == 0:
                e[i] = self.w[i] - 0
            else:
                e[i] = self.w[i] - y[i-1]  # 计算当前时刻的误差
            u[i] = pid.compute_control(self.t[:i+1], e[:i+1])  # 计算当前时刻的控制量
            y[i] = self.sys.simulate(self.t[:i+1], u[:i+1])  # 计算当前时刻的输出

        pd.DataFrame({'t': self.t, 'y': y, 'u': u, 'e': e}).to_csv('data.csv', index=False)

        ise = np.sum(e ** 2)  # 计算 ISE
        overshoot = np.max(y) - self.w[-1] if np.max(y) > self.w[-1] else 0  # 计算超调量
        rise_time = next((i for i, value in enumerate(y) if value >= self.w[-1]), len(self.t)) - self.t[0]  # 计算上升时间
        cost = ise + overshoot + rise_time  # 计算总代价
        return cost

    def run_optimization(self, optimizer):
        best_solution, best_cost = optimizer.optimize(self.cost_func)  # 进行优化
        return best_solution, best_cost

# 系统参数
num1 = [2.171]
den1 = [1, 0.1512]
num2 = [0.0781, 9.636e-4]
den2 = [1, 19.38, 0.5134]
delay = 15
t = np.arange(0,1000,1)
w = 6 * np.ones_like(t)

# 创建系统模型
sys = SystemModel(num1, den1, num2, den2, delay)

# 初始化蚁群优化算法
optimizer = Optimizer(num_ants=50, num_iterations=10, alpha=1, beta=1, rho=0.1, q=1,
                      param_ranges=[np.linspace(0, 3, 50), np.linspace(0, 1, 50), np.linspace(0, 1, 50)])

# 运行模拟和优化
simulation = Simulation(sys, w, t)
best_solution, best_cost = simulation.run_optimization(optimizer)

print("Best solution: ", best_solution)
print("Best cost: ", best_cost)

# 使用最优解计算 PID 控制器的输出
pid = PIDController(*best_solution)
u = pid.compute_control(t, simulation.e)
y = sys.simulate(t, u)

# 绘制结果
plt.figure()
plt.plot(t, y, label='PID Controller Output')
plt.plot(t, w, label='Expected Output')
plt.legend()
plt.show()
