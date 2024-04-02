import numpy as np
import matplotlib.pyplot as plt
import control as ct

num1 = [2.171]
den1 = [1, 0.1512]
num2 = [0.0781, 9.636e-4]
den2 = [1, 19.38, 0.5134]

sys1 = ct.tf(num1, den1,inputs='u', outputs='y')
sys2 = ct.tf(num2, den2,inputs='u', outputs='y')
plantcont = ct.series(sys1, sys2)
t, y = ct.step_response(plantcont,250)
plt.plot(t, y, label='continouous-time model')

simulation_dt = 2
plant_simulator = ct.c2d(plantcont, simulation_dt, method='zoh')
t, y = ct.step_response(plant_simulator,250)

plt.ion()
PID = ct.rootlocus_pid_designer(plantcont, gain='D', sign=6, input_signal='r')

plt.show(block=True)