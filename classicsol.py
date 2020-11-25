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


def stokes_regime(initial,times,drag_coeff,lin_coeff):
    def force(x,v):
        return -drag_coeff*v - lin_coeff*x
    sol = ODE_solver(initial,force,times)
    plt.plot(sol.get_times(),sol.get_positions(),marker = '')
    plt.show()

def antistokes_regime(initial,times,drag_coeff,lin_coeff,cub_coeff):
    def force(x,v):
        return drag_coeff*v - lin_coeff*x - cub_coeff*(x**3)
    sol = ODE_solver(initial,force,times)
    return sol


def antistokes_regime_cub_plot(initial,times,drag_coeff,lin_coeff,cub_coeffs):
    fig, ax = plt.subplots(len(cub_coeffs),sharex = True)
    for i in range(len(cub_coeffs)):
        cub_coeff = cub_coeffs[i]
        sol = antistokes_regime(initial,times,drag_coeff,lin_coeff,cub_coeff)
        ax[i].plot(sol.get_times(),sol.get_positions(),marker = '')
        ax[i].set_title('cub_coeff = {}'.format(cub_coeff))
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.show()
    fig.tight_layout()




initial = [1,0]
times = np.linspace(0,100,1000)

'''
solve = ODE_solver(initial,force,times)
plt.plot(solve.get_times(),solve.get_positions(),marker = '')
plt.show()
'''

cub_coeffs = [100,50,20,10]

x = antistokes_regime_cub_plot(initial,times,0.05,1,cub_coeffs)





