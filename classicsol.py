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
        sol = scipy.integrate.solve_ivp(self.ODE,(t_init,t_fin),self.y0,t_eval = self.time_eval,rtol = 1e-12)
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


def stokes_regime(initial,times,const_coeff,drag_coeff,lin_coeff,cub_coeff):
    def force(x,v):
        return  - const_coeff -drag_coeff*v - lin_coeff*x - cub_coeff*(x**3)
    sol = ODE_solver(initial,force,times)
    return sol

def antistokes_regime(initial,times,const_coeff,drag_coeff,lin_coeff,cub_coeff):
    def force(x,v):
        return -const_coeff + drag_coeff*v - lin_coeff*x - cub_coeff*(x**3)
    sol = ODE_solver(initial,force,times)
    return sol

def non_linear_regime(initial,times,const_coeff,quad_coeff,lin_coeff,cub_coeff):
    def force(x,v):
        return - const_coeff + quad_coeff*(v**2) - lin_coeff*(x) - cub_coeff*(x**3)
    sol = ODE_solver(initial,force,times)
    return sol

def stokes_regime_position_plot(initial,times,drag_coeff,const_coeff,lin_coeff,cub_coeff):
    fig, ax = plt.subplots()
    sol = stokes_regime(initial,times,drag_coeff,const_coeff,lin_coeff,cub_coeff)
    positions = sol.get_positions()
    plt.plot(times,positions)
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Plot of the electron position against time using a cub coeff of 100')
    plt.show()
    return fig
def stokes_regime_cub_coeff_position_ft_plot(initial,times,drag_coeff,const_coeff,lin_coeff,cub_coeffs):
    fig,ax = plt.subplots()
    for i in range(len(cub_coeffs)):
        cub_coeff = cub_coeffs[i]
        sol = stokes_regime(initial,times,drag_coeff,const_coeff,lin_coeff,cub_coeff)
        positions = sol.get_positions()
        position_ave = np.average(positions)
        positions_ft = np.fft.rfft(positions - position_ave)
        abs_positions_ft = np.abs(positions_ft)
        frequencies = (2*np.pi/times[-1])*range(len(abs_positions_ft))
        plt.plot(frequencies,abs_positions_ft,label = 'cub_coeff = {}'.format(cub_coeff))
    plt.xlabel('Frequency')
    plt.ylabel('abs(FT(positions))')
    plt.title('Plot of the FT of the electron position')
    plt.legend()
    plt.show()
    return fig

def stokes_regime_cub_coeff_position_plot(initial,times,drag_coeff,const_coeff,lin_coeff,cub_coeffs):
    fig,ax = plt.subplots()
    for i in range(len(cub_coeffs)):
        sol = stokes_regime(initial,times,drag_coeff,const_coeff,lin_coeff,cub_coeffs[i])
        positions = sol.get_positions()
        plt.plot(times,positions,label = 'cub_coeff = {}'.format(cub_coeffs[i]))
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Position plot for different cubic coefficients in the restoring force')
    plt.legend()
    plt.show()
    return fig

def stokes_regime_energy_plot(initial,times,const_coeff,drag_coeff,lin_coeff,cub_coeff):
    fig,ax = plt.subplots()
    sol = stokes_regime(initial,times,const_coeff,drag_coeff,lin_coeff,cub_coeff)
    positions = sol.get_positions()
    velocities = sol.get_velocities()
    energy = (1/2)*(velocities**2) + (1/2)*(lin_coeff)*(positions**2) + (1/4)*(cub_coeff)*(positions**4)
    plt.plot(times,energy)
    plt.show()
    return fig

def stokes_regime_energy_ft_plot(initial,times,const_coeff,drag_coeff,lin_coeff,cub_coeff):
    fig,ax = plt.subplots()
    sol = stokes_regime(initial,times,drag_coeff,const_coeff,lin_coeff,cub_coeff)
    positions = sol.get_positions()
    velocities = sol.get_velocities()
    energy = (1/2)*(velocities**2) + (1/2)*(lin_coeff)*(positions**2) + (1/4)*(cub_coeff)*(positions**4)
    energy_ave = np.average(energy)
    energy_ft = np.fft.rfft(energy - energy_ave)
    abs_energy_ft = abs(energy_ft)
    plt.plot((2*np.pi/times[-1])*range(len(abs_energy_ft)),abs_energy_ft)
    plt.show()
    return fig

def stokes_regime_cub_coeff_energy_ft_plot(initial,times,const_coeff,drag_coeff,lin_coeff,cub_coeffs):
    fig,ax = plt.subplots()
    for i in range(len(cub_coeffs)):
        sol = stokes_regime(initial,times,const_coeff,drag_coeff,lin_coeff,cub_coeffs[i])
        positions = sol.get_positions()
        velocities = sol.get_velocities()
        energy = (1/2)*(velocities**2) + (1/2)*(lin_coeff)*(positions**2) + (1/4)*(cub_coeffs[i])*(positions**4)
        energy_ave = np.average(energy)
        energy_ft = np.fft.rfft(energy - energy_ave)
        abs_energy_ft = abs(energy_ft)
        plt.plot((2*np.pi/times[-1])*range(len(abs_energy_ft)),abs_energy_ft,label = 'Cub_coeff = {}'.format(cub_coeffs[i]))
    plt.xlabel('Frequency')
    plt.ylabel('abs(FT(Energy))')
    plt.legend()
    plt.title('FT of the energy for different cubic coefficients in the restoring force (const_coeff = {} drag_coeff = {})'.format(const_coeff,drag_coeff))
    plt.show()
    return fig

def stokes_regime_cub_coeff_frequency_plot(initial,times,const_coeff,drag_coeff,lin_coeff,cub_coeffs):
    fig,ax = plt.subplots()
    ss_frequencies = []
    for i in range(len(cub_coeffs)):
        cub_coeff = cub_coeffs[i]
        sol = stokes_regime(initial,times,const_coeff,drag_coeff,lin_coeff,cub_coeff)
        positions = sol.get_positions()
        velocities = sol.get_velocities()
        energy = (1/2)*(velocities**2) + (1/2)*(lin_coeff)*(positions**2) + (1/4)*(cub_coeff)*(positions**4)
        energy_ave = np.average(energy)
        energy_ft = np.fft.rfft(energy - energy_ave)
        abs_energy_ft = abs(energy_ft)
        frequencies = (2*np.pi/times[-1])*range(len(abs_energy_ft))
        ss_frequency = frequencies[np.argmax(abs_energy_ft)]
        ss_frequencies.append(ss_frequency)
    plt.plot(cub_coeffs,ss_frequencies,marker = 'x',linestyle = '')
    plt.xlabel('cub_coeff')
    plt.ylabel('Peak frequency in energy spectrum')
    plt.title('Plot of the peak frequency in the energy spectrum for different cub_coeffs')
    plt.show()
    return fig


def non_linear_regime_position_plot(initial,times,const_coeff,quad_coeff,lin_coeff,cub_coeff):
    sol = non_linear_regime(initial,times,const_coeff,quad_coeff,lin_coeff,cub_coeff)
    positions = sol.get_positions()
    plt.plot(times,positions)
    plt.show()
    return fig

def non_linear_regime_ft_plot(initial,times,const_coeff,quad_coeff,lin_coeff,cub_coeff):
    sol = non_linear_regime(initial,times,const_coeff,quad_coeff,lin_coeff,cub_coeff)
    positions_ss = sol.get_positions()[int(len(sol.get_positions())/2):]
    ft_ss = np.fft.rfft(positions_ss)
    abs_ft_ss = abs(ft_ss)
    plt.plot((1/times[-1])*range(len(abs_ft_ss)),abs_ft_ss)
    plt.show()
    return fig


def antistokes_regime_cub_coeff_plot(initial,times,const_coeff,drag_coeff,lin_coeff,cub_coeffs):
    fig, ax = plt.subplots(len(cub_coeffs),sharex = True)
    for i in range(len(cub_coeffs)):
        cub_coeff = cub_coeffs[i]
        sol = antistokes_regime(initial,times,const_coeff,drag_coeff,lin_coeff,cub_coeff)
        if len(cub_coeffs) == 1:
            ax.plot(sol.get_times(),sol.get_positions(),label = ' x = {}'.format(cub_coeff))
        else:
            ax[i].plot(sol.get_times(),sol.get_positions(),label = ' x = {}'.format(cub_coeff))
    fig.legend(loc = 'center right')
    fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
    plt.xlabel('Time/AU')
    plt.ylabel('Position/AU')
    plt.title('Electron positions for parameters ({},{},{},x)'.format(const_coeff,drag_coeff,lin_coeff))
    plt.show()
    return fig

def antistokes_regime_correlation_plot(initial,times,drag_coeff,const_coeff,lin_coeff,cub_coeff):
    sol = antistokes_regime(initial,times,drag_coeff,const_coeff,lin_coeff,cub_coeff)
    correlation = np.correlate(sol.get_positions(),sol.get_positions(),'full')[sol.get_number_of_times()-1:]
    plt.plot(sol.get_times(),correlation)
    plt.show()
    return fig

def antistokes_regime_energy_fourier_transform_plot(initial,times,drag_coeff,const_coeff,lin_coeff,cub_coeff):
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

def antistokes_regime_power_spectrum_peak_plot(initial,times,drag_coeff,const_coeff,lin_coeffs,cub_coeff):
    peak_frequencies = []
    for i in range(len(lin_coeffs)):
        sol = antistokes_regime(initial,times,drag_coeff,const_coeff,lin_coeffs[i],cub_coeff)
        ft_full = np.fft.rfft(sol.get_positions())
        abs_ft_full = np.abs(ft_full)
        peak_frequency = np.argmax(abs_ft_full)
        peak_frequencies.append(peak_frequency)
    plt.plot(lin_coeffs,peak_frequencies,marker = '+')
    plt.show()
    return fig
    
def antistokes_regime_energy_plot(initial,times,drag_coeff,const_coeff,lin_coeff,cub_coeff):
    sol = antistokes_regime(initial,times,drag_coeff,const_coeff,lin_coeff,cub_coeff)
    positions = sol.get_positions()
    velocities = sol.get_velocities()
    energy = (1/2)*(velocities**2) + (1/2)*(lin_coeff)*(positions**2) + (1/4)*(cub_coeff)*(positions**4)
    plt.plot(times,energy)
    plt.show()
    return fig

def antistokes_regime_energy_ft_plot(initial,times,const_coeff,drag_coeff,lin_coeff,cub_coeff):
    fig,ax = plt.subplots()
    sol = antistokes_regime(initial,times,drag_coeff,const_coeff,lin_coeff,cub_coeff)
    positions = sol.get_positions()
    velocities = sol.get_velocities()
    energy = (1/2)*(velocities**2) + (1/2)*(lin_coeff)*(positions**2) + (1/4)*(cub_coeff)*(positions**4)
    energy_ave = np.average(energy)
    energy_ft = np.fft.rfft(energy - energy_ave)
    abs_energy_ft = abs(energy_ft)
    plt.plot((2*np.pi/times[-1])*range(len(abs_energy_ft)),abs_energy_ft)
    plt.show()
    return fig



initial = [1,0]
times = np.linspace(0,100,10000)

#z = antistokes_regime_power_spectrum_peak_plot(initial,times,0.05,1,[1,2,3],10)
#a = antistokes_regime_energy_plot(initial,times,0.5,1,1,10)
#b = stokes_regime_cub_coeff_frequency_plot(initial,times,0,0.5,4,np.linspace(0,500,50))
#b = stokes_regime_energy_plot(initial,times,0,0.5,4,5)
#b = stokes_regime_cub_coeff_energy_ft_plot(initial,times,0,0.5,4,[0,1,2,3,4])
#c = stokes_regime_position_plot(initial,times,0,0.5,4,5)
#c = stokes_regime_cub_coeff_position_ft_plot(initial,times,0,0.5,4,[0,1,2,3,10])
#x = antistokes_regime_fourier_transform_plot(initial,times,0.05,1,1,10)
#x.savefig('Antistokes regime FT plot for a time length of 800',dpi = 300,bbox_inches = 'tight')
y = antistokes_regime_cub_coeff_plot(initial,times,0,0.01,4,[1,1000])
#y.savefig('Antistokes regime displacement plots for different cubic coefficients sampled 5000 times')
#d = non_linear_regime_position_plot(initial,times,1,0.5,1,0.1)
#e = non_linear_regime_ft_plot(initial,times,1,0.5,1,0.5)
