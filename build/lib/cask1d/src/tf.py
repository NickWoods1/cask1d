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

    # Initialise chemical potential (Lagrange multiplier in order to keep int(rho) = N)
    chem_potential = 0

    # Solve the integral TF equations (e.g. Lieb (1977) and Parr DFT book)
    j, error = 1, 1
    while error > params.tol_tf:

        # Evaluate the residual used to solve the TF equations (should be zero...)
        residual = tf_objective_function(params, density, chem_potential)

        # Update density w/ steepest descent
        density += params.step_length*residual

        # Update chemical potential w/ steepest descent (modify chem_pot s.t. constraint is satisfied)
        chem_potential -= params.step_length*(np.sum(density)*params.dx - params.num_electrons)

        # Plot density as iterations progress...
        #if j % 20 == 0:
        #    plt.plot(density)
        #    plt.pause(0.01)
        #    plt.clf()

        # Ensure no negative density regions
        for i in range(0,params.Nspace):
            if density[i] < 0:
                density[i] = 0

        # Error (L1 norm)
        error = np.sum(abs(residual))
        print('Error = {0} at iteration {1}. Total energy = {2}'.
              format(error, j, tf_energy_functional(params, density)))

        j += 1

    print('Final energy = {0} with chemical potential {1}'.format(
            tf_energy_functional(params, density), chem_potential))

    plt.plot(initial_guess_density(params))
    plt.plot(density)
    plt.show()

    return density


def tf_objective_function(params, density, chem_potential):
    """
    Implements TF equation = residual != 0 (https://physics.nyu.edu/LarrySpruch/Lieb.pdf)
    """

    # TF constant
    C = 0.5*(3*np.pi**2)**(2/3)

    # TF equation
    residual = - v_ext(params) - v_h(params, density) - C*density**(2/3) + chem_potential + (6 / np.pi)**(1/3)*density**(1/3)

    return residual


def tf_energy_functional(params, density):
    """
    TF energy functional E[rho] = K_TF + E_ext + E_h
    """

    # TF constant
    C = (3/10)*(3*np.pi**2)**(2/3)

    # Kinetic energy
    kinetic_energy = C*np.sum(density**(5/3))*params.dx

    # Hartree energy
    hartree_pot = v_h(params, density)
    hartree_energy = np.sum(density*hartree_pot)*params.dx

    # External
    external_pot = v_ext(params)
    external_energy = np.sum(density*external_pot)*params.dx

    return kinetic_energy + hartree_energy + external_energy
