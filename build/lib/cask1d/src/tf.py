import numpy as np
import scipy as sp
from scipy.optimize import root
from cask1d.src.dft import initial_guess_density
from cask1d.src.hamiltonian import construct_hamiltonian_independent, v_ext, v_h
import matplotlib.pyplot as plt


"""
Thomas-Fermi Theory
"""

def minimise_energy_tf(params):

    # Generate initial guess density (sum weighted Gaussians)
    density = initial_guess_density(params)

    # Compute the ground-state potential using a scipy optimiser
    opt_info = root(tf_objective_function, density, args=(params), method='anderson', tol=1e-15)

    plt.plot(opt_info.x)
    plt.show()

    # Output v_ks

def tf_objective_function(density, params):

    gamma = 0.5*(3 * np.pi**2)**(2/3)
    vext = v_ext(params)
    vh = v_h(params, density)
    tf = np.zeros(params.Nspace)

    plt.plot(vext-vh)
    plt.plot(vh)
    plt.plot(vext)
    plt.show()

    for i in range(params.Nspace):

        if (vext[i] - vh[i]) > 0:
            tf[i] = vext[i] - vh[i] - gamma*density[i]**(2/3)
        else:
            tf[i] = -gamma*density[i]**(2/3)

    print(np.sum(abs(tf)))

    return tf
