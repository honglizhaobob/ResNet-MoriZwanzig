# File containing all test ODEs considered
import numpy as np

def linear_system(y, t, alpha):
    """ 
        a simple 2d linear dynamical system. 
        

        See:
            Xiaohan et al., Learning Reduced Systems via Deep Neural Networks with Memory

    """
    dydt = np.zeros(2)
    dydt[0] = y[0] - 4*y[1]
    dydt[1] = 4*y[0] - alpha*y[1]
    return dydt

def nonlinear_system(y, t, alpha=0.1, beta=8.91):
    """
        A 2d nonlinear dynamical system.

        See:
            Xiaohan et al., Learning Reduced Systems via Deep Neural Networks with Memory
    """
    dydt = np.zeros(2)
    dydt[0] = y[1]
    dydt[1] = -alpha*y[1] - beta*np.sin(y[0])
    return dydt

def van_der_pol(y, t, mu=1000):
    """
        The two-dimensional formulation of Van der Pol oscillator with
        mu as the parameter. See example in:

        https://www.mathworks.com/help/matlab/math/solve-stiff-odes.html
    """
    dydt = np.zeros(2)
    dydt[0] = y[1]
    dydt[1] = mu*(1-y[0]**2)*y[1]-y[0]
    return dydt







