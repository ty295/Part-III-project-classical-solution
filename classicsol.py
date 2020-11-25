import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt

class ODE_solver():

    def __init__(self,y0,force_eqn,time_eval):
        self.y0 = y0
        self.force_eqn = force_eqn
        self.time_eval = time_eval
        self.solver()

    def solver(self):
        t_init = self.time_eval[0]
        t_fin = self.time_eval[-1]
        sol = scipy.integrate.solve_ivp(self.ODE,(t_init,t_fin),self.y0,t_eval = self.time_eval)
        self.x = sol.y[0]
        self.v = sol.y[1]

    def ODE(self,t,y):
        x,v = y
        dydt = [v,self.force_eqn(x,v)]
        return  dydt

    def get_positions(self):
        return self.x

    def get_velocities(self):
        return self.v

    def get_times(self):
        return self.time_eval



initial = [1,0]
times = np.linspace(0,20,1000)

def force(x,v):
    return -v - 5*x


solve = ODE_solver(initial,force,times)
plt.plot(solve.get_times(),solve.get_positions(),marker = '')
plt.show()








