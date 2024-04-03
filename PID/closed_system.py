import numpy as np
import matplotlib.pyplot as plt
import control as ct

class PID:
    def __init__(self, kp, ki, kd, N=10):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.N = N

    def _build_transfer_function(self):
        ti = self.kp/self.ki
        td = self.kd/self.kp

        Gc = self.kp*ct.tf([ti*td*(self.N+1), self.N*ti+td, self.N], [ti*td, self.N*ti, 0])
        return Gc
    
    def __repr__(self):
        return f'kp={self.kp}, ki={self.ki}, kd={self.kd}, N={self.N}'
    
    def __call__(self):
        return self._build_transfer_function()
class Output:
    # 系统传递函数，频率->流量
    NUM1 = [2.171]
    DEN1 = [1, 0.1512]

    # 系统传递函数，流量->温度
    NUM2 = [0.0781, 9.636e-4]
    DEN2 = [1, 19.38, 0.5134]

    def __init__(self):
        pass

    def _build_transfer_function(self):
        G1 = ct.tf(Output.NUM1, Output.DEN1)
        G2 = ct.tf(Output.NUM2, Output.DEN2)
        self.Gy = G1*G2
        return self.Gy
    
    def __repr__(self):
        return print(f'Gy={self.Gy}' if self.Gy is not None else 'None')
    
    def __call__(self):
        return self._build_transfer_function()
    
class Delay:
    # 测量延迟时间
    T_DELAY = 15
    NORDER = 1

    def __init__(self, t_delay = None, norder=None):
        if t_delay is not None:
            self.T_DELAY = t_delay
        if norder is not None:
            self.NORDER = norder
    
    def _build_transfer_function(self):
        num, den = ct.pade(self.T_DELAY, self.NORDER)
        self.Gd = ct.tf(num, den)
        return self.Gd
    
    def __repr__(self):
        return print(f'Gd={self.Gd}' if self.Gd is not None else 'None')

    def __call__(self):
        return self._build_transfer_function()
    
class Close_Plant:
    def __init__(self,pid, output, delay):
        self.Gc = pid()
        self.Gy = output()
        self.Gd = delay()
    
    def _build_transfer_function(self):
        Gc = self.Gc
        Gy = self.Gy
        Gd = self.Gd

        self.Hyr = Gc*Gy/(1+Gc*Gy*Gd)
        return self.Hyr
    
    def __repr__(self):
        return f'Hyr={self.Hyr}' if self.Hyr is not None else 'None'
    
    def __call__(self):
        return self._build_transfer_function()
    
class Simulation:
    def __init__(self, plant,t_final=450):
        self.plant = plant()
        self.t_final = t_final
    
    def _build_transfer_function(self):
        Hyr = self.plant
        t = np.linspace(0, self.t_final, 1000)
        t, y = ct.step_response(Hyr, T=t)
        s = ct.step_info(y,T=t) # 获取阶跃响应的性能数据
        return t, y, s
    
    def __repr__(self):
        return f'plant={self.plant}'
    
    def __call__(self):
        return self._build_transfer_function()

def main(kp, ki, kd, N):
    pid = PID(kp, ki, kd, N)
    output = Output()
    delay = Delay()
    plant = Close_Plant(pid, output, delay)
    simulation = Simulation(plant)
    t, y, s = simulation()
    return t, y, s

if __name__ == '__main__':
    kp = 19
    ki = 1.2
    kd = 13
    N = 5.08
    t, y, s = main(kp, ki, kd, N)
    plt.plot(t, y)
    plt.show()
