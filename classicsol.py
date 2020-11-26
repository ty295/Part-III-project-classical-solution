import scipy.integrate
import scipy.fft as fft
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
    
    def get_number_of_times(self):
        return self.time_eval.size


def stokes_regime(initial,times,drag_coeff,lin_coeff):
    def force(x,v):
        return -drag_coeff*v - lin_coeff*x
    sol = ODE_solver(initial,force,times)
    plt.plot(sol.get_times(),sol.get_positions(),marker = '')
    plt.show()

def antistokes_regime(initial,times,drag_coeff,const_coeff,lin_coeff,cub_coeff):
    def force(x,v):
        return -const_coeff + drag_coeff*v - lin_coeff*x - cub_coeff*(x**3)
    sol = ODE_solver(initial,force,times)
    return sol


def antistokes_regime_cub_plot(initial,times,drag_coeff,const_coeff,lin_coeff,cub_coeffs):
    fig, ax = plt.subplots(len(cub_coeffs),sharex = True)
    colours = ['r','g']
    for i in range(len(cub_coeffs)):
        cub_coeff = cub_coeffs[i]
        sol = antistokes_regime(initial,times,drag_coeff,const_coeff,lin_coeff,cub_coeff)
        if len(cub_coeffs) == 1:
            ax.plot(sol.get_times(),sol.get_positions(),colours[i],label = ' x = {}'.format(cub_coeff))
        else:
            ax[i].plot(sol.get_times(),sol.get_positions(),colours[i],label = ' x = {}'.format(cub_coeff))
    fig.legend(loc = 'center right')
    fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
    plt.xlabel('Time/AU')
    plt.ylabel('Position/AU')
    plt.title('Electron positions for parameters ({},{},{},x) sampled {} times'.format(drag_coeff,const_coeff,lin_coeff,len(times)))
    plt.show()
    return fig

def antistokes_regime_correlation_plot(initial,times,drag_coeff,const_coeff,lin_coeff,cub_coeff):
    sol = antistokes_regime(initial,times,drag_coeff,const_coeff,lin_coeff,cub_coeff)
    correlation = np.correlate(sol.get_positions(),sol.get_positions(),'full')[sol.get_number_of_times()-1:]
    plt.plot(sol.get_times(),correlation)
    plt.show()
    return fig

def antistokes_regime_fourier_transform_plot(initial,times,drag_coeff,const_coeff,lin_coeff,cub_coeff):
    fig,ax = plt.subplots(2)
    sol = antistokes_regime(initial,times,drag_coeff,const_coeff,lin_coeff,cub_coeff)
    ft_full = np.fft.rfft(sol.get_positions())
    ft_ss = np.fft.rfft(sol.get_positions()[int(len(sol.get_positions())*0.5):])
    abs_ft_full = np.abs(ft_full)
    abs_ft_ss = np.abs(ft_ss)
    ax[0].plot((1/times[-1])*range(len(abs_ft_full)),abs_ft_full,'r')
    ax[1].plot((1/times[-1])*range(len(abs_ft_ss)),abs_ft_ss,'g')
    ax[1].set_xlabel('Frequency/AU')
    fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
    plt.ylabel('abs(FT)/AU')
    fig.legend(labels = ['Full solution','Steady state solution'],loc = 'center right')
    plt.title('FT of the electron position for parameters ({},{},{},{}) sampled for a time length of {}'.format(drag_coeff,const_coeff,lin_coeff,cub_coeff,times[-1]))
    fig.tight_layout()
    plt.show()
    return fig
    





initial = [1,0]
times = np.linspace(0,500,5000)

x = antistokes_regime_fourier_transform_plot(initial,times,0.05,1,1,100)
#x.savefig('Antistokes regime FT plot for a time length of 800',dpi = 300,bbox_inches = 'tight')
#y = antistokes_regime_cub_plot(initial,times,0.05,1,1,[3,1000])
#y.savefig('Antistokes regime displacement plots for different cubic coefficients sampled 5000 times')


