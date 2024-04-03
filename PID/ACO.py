import numpy as np
from closed_system import *

def system_caculate(kp, ki, kd, N):
    pid = PID(kp, ki, kd, N)
    output = Output()
    delay = Delay()
    plant = Close_Plant(pid, output, delay)
    simulation = Simulation(plant)
    respond = simulation()
    return respond

def evaluate(respond):
    t,y,s = respond
    error = np.abs(1 - y)
    iae = np.trapz(error, t)
    w1,w2,w3 = 1,1,2
    rise_time = s['RiseTime']
    overshoot = s['Overshoot']
    settling_time = s['SettlingTime']
    result = w1*iae + w2*rise_time
    return result

def result_show(t,y):
    plt.plot(t, y)
    plt.show()

def data_processing(data):
    pass

class AntColonyOptimization:
    def __init__(self, obj_func:system_caculate, n_ants:int, n_iter:int, alpha, beta, evaporation_rate, Q, init_var_ranges):
        self.obj_func = obj_func
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.n_params = len(init_var_ranges)
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.init_var_ranges = init_var_ranges
        self.best_score = np.inf
        self.best_params = None
    
    def optimize(self):
        pheromone = np.ones((self.n_ants, self.n_params)) # 初始化信息素，均为1

        params = np.zeros((self.n_ants, self.n_params))
        for i in range(self.n_ants):
            params[i] = self.generate_init_params()  # 生成参数

        for _ in range(self.n_iter):

            scores = np.zeros((self.n_ants,))
            for i in range(self.n_ants):
                scores[i] = evaluate(self.obj_func(*params[i]))

            self.get_params(scores,params)

            pheromone *= self.evaporation_rate
            for j in range(self.n_params):
                pheromone[:,j] += self.Q/scores # 更新信息素

            probabilitis = np.zeros((self.n_ants, self.n_params))
            for i in range(self.n_ants):
                probabilitis[i] = pheromone[i]**self.alpha / np.sum(scores)**self.beta # 使用结果和信息素更新概率
            
            for j in range(self.n_params):
                probabilitis[:,j] /= np.sum(probabilitis[:,j])

            new_params = np.zeros_like(params)
            for i in range(self.n_ants):
                for j in range(self.n_params):
                    new_params[i,j] = np.random.choice(params[:,j], p=probabilitis[:,j])
            params = new_params

        self.best_score_info = self.obj_func(*self.best_params)[2]

    def get_params(self,scores,params):
        result_score = np.min(scores)
        result_params = params[np.argmin(scores)]

        if result_score < self.best_score:
            self.best_score = result_score
            self.best_params = result_params

    def generate_init_params(self):
        params = np.zeros(self.n_params)
        for i in range(self.n_params):
            params[i] = np.random.uniform(self.init_var_ranges[i][0], self.init_var_ranges[i][1])
        return params


if __name__ == '__main__':
    n_ants = 50
    n_iter = 100
    alpha = 1
    beta = 1
    evaporation_rate = 0.5
    Q = 100
    init_var_ranges = [(10, 20), (0, 1), (10, 40), (3, 10)]

    aco = AntColonyOptimization(system_caculate, n_ants, n_iter, alpha, beta, evaporation_rate, Q, init_var_ranges)
    aco.optimize()
    print(aco.best_params, aco.best_score, aco.best_score_info)
    t,y,_ = system_caculate(*aco.best_params)
    result_show(t,y)