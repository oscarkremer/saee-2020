import matplotlib
import numpy as np
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
matplotlib.rc('axes', titlesize=15)
matplotlib.rc('axes', labelsize=15)
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
from scipy.optimize import differential_evolution


class SpringMass:
    '''
    Classe sobre sistemas 
    massa mola
    '''
    def __init__(self, K, B, M):
        self.K = K
        self.B = B
        self.M = M


def state_equations(t, x, sys, gains):
    '''
    Implementa a equação na 
    forma de variaveis de estado
    do sistema físico considerado.
    t: instante em um tempo específico
    x: vetor de estados do sistema
    sys: sistema massa mola
    gains: ganhos proporcional e derivativo
    '''
    x1 = x[0]
    x2 = x[1]
    xd = np.sin(t)
    xdponto = np.cos(t)
    u = gains['proporcional']*(xd-x1) + gains['derivative']*(xdponto-x2)
    dx1dt=x2; 
    if abs(x2) > 0.01:
        dx2dt= (1/sys.M)*(u - sys.B*x2 - sys.K*x1 - sys.M*9.8*0.3)
    else:
        dx2dt= (1/sys.M)*(u - sys.B*x2 - sys.K*x1 - sys.M*9.8*0.8)
    return np.array([dx1dt,dx2dt])

def euler_solver(x, times, delta_t, sys, gains):
    '''
    Função que resolve a equação
    diferencial por Euler, fazendo a
    acumulação dos estados calculados com
    state_equation.
    x: estados iniciais
    times: vetor de tempo
    delta_t: discretização do tempo
    sys:sistema físico considerado
    gains: ganhos proporcional e derivativo
    '''
    x_plot = []
    for time in times:
        x = x + delta_t*state_equations(time, x, sys, gains)
        x_plot.append(x)
    return np.array(x_plot)

def itae(gains):
    '''
    Função que calcula o 
    ITAE(integral time absolute error)
    para um determinado caso de tempo,
    valor de referência e sistema 
    massa mola.
    gains
    '''
    delta_t = 1/100
    t = np.arange(0, 10, delta_t)
    x0 = np.array([0, 0])
    xd = np.sin(t)
    control_gains = {'proporcional': gains[0], 'derivative': gains[1]}
    sys = SpringMass(100, 8, 10)
    x1 = euler_solver(x0, t, delta_t, sys, control_gains)[:,0]
    if np.isnan(x1[-1]):
        return 10**6
    else:
        return sum(t*abs(xd-x1))


if __name__=='__main__':    
    bounds = [(0.0001, 10000), (0.001, 10000)]
    result = differential_evolution(itae, bounds, maxiter=100, popsize=30, disp=True)
    print(result.x)
    print(result.fun)
    print(itae([1000, 10]))
    print(itae([1000, 200]))
    print(itae([200, 1000]))
    print(itae([10, 1000]))
