import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from density2potential.utils.physics import calculate_density_ks
from density2potential.utils.math import discrete_Laplace, normalise_function, norm
import scipy.linalg as linalg
from cask1d.src.hamiltonian import construct_hamiltonian_independent, update_hamiltonian
from cask1d.src.hamiltonian import evaluate_energy_functional, v_ext
from cask1d.src.scf import pulay_mixing, newton_mixing, pulay_mixing2
from cask1d.src.linear_response import calculate_susceptibility, calculate_dielectric

"""
Computes the self-consistent Kohn-Sham orbitals, density, and energy given a density functional and external
potential 
"""


def minimise_energy_dft(params):

    #precond = np.load('precond.npy')

    # Array that will store SCF iterative densities and residuals
    history_of_densities_in = np.zeros((params.history_length, params.Nspace))
    history_of_densities_out = np.zeros((params.history_length, params.Nspace))
    history_of_residuals = np.zeros((params.history_length, params.Nspace))
    density_differences = np.zeros((params.history_length, params.Nspace))
    residual_differences = np.zeros((params.history_length, params.Nspace))

    # Generate initial guess density (sum weighted Gaussians)
    density_in = initial_guess_density_nonint(params)

    # Construct the independent part of the hamiltonian, i.e. KE + v_external
    hamiltonian_independent = construct_hamiltonian_independent(params)

    # SCF loop
    i, error = 0, 1
    while error > params.tol_ks:

        # Iteration number modulus history length
        i_mod = i % params.history_length
        i_mod_prev = (i-1) % params.history_length

        hamiltonian = update_hamiltonian(params, hamiltonian_independent, density_in)

        # Solve H psi = E psi
        eigenvalues, eigenvectors = linalg.eigh(hamiltonian)
        eigenvectors = normalise_function(params, eigenvectors[:,0:])


        # Extract lowest lying num_particles eigenfunctions and normalise
        wavefunctions_ks = eigenvectors[:,0:params.num_particles]
        wavefunctions_ks[:,0:params.num_particles] = normalise_function(params, wavefunctions_ks[:,0:params.num_particles])

        # Calculate the output density
        density_out = calculate_density_ks(params, wavefunctions_ks)



        susceptibility = calculate_susceptibility(params, eigenvectors, eigenvalues)
        dielectric = calculate_dielectric(params, density_out, susceptibility)
        precond = dielectric
        precond = np.eye(params.Nspace)


        # Calculate total energy
        total_energy = evaluate_energy_functional(params, wavefunctions_ks, density_out)

        # L1 error between input and output densities
        error = np.sum(abs(density_in - density_out)*params.dx)
        print('SCF error = {0} at iteration {1} with energy {2}'.format(error, i, total_energy))

        # Store densities/residuals within the iterative history data
        history_of_densities_in[i_mod,:] = density_in
        history_of_densities_out[i_mod,:] = density_out
        history_of_residuals[i_mod,:] = density_out - density_in

        if i == 0:
            # Damped linear step for the first iteration
            #density_in = density_in - params.step_length * (density_in - density_out)
            density_in = density_in - params.step_length * np.dot(precond,(density_in - density_out))

        elif i > 0:
            # Store more iterative history data...
            density_differences[i_mod_prev] = history_of_densities_in[i_mod] - history_of_densities_in[i_mod_prev]
            residual_differences[i_mod_prev] = history_of_residuals[i_mod] - history_of_residuals[i_mod_prev]

            # Perform Pulay step using the iterative history data
            density_in = pulay_mixing2(params, density_differences, residual_differences,
                                     history_of_residuals[i_mod], history_of_densities_in[i_mod], i, precond)

            #density_in = newton_mixing(params, history_of_densities_in[i_mod], history_of_residuals[i_mod], eigenvectors, eigenvalues)

        i += 1

    # Plot linear response functions of the converged system
    eigenvalues, eigenvectors = linalg.eigh(hamiltonian)
    eigenvectors = normalise_function(params, eigenvectors[:,0:])
    susceptibility = calculate_susceptibility(params, eigenvectors, eigenvalues)
    dielectric = calculate_dielectric(params, density_out, susceptibility)
    plt.imshow(susceptibility.real, origin='lower')
    plt.show()
    plt.clf()
    plt.imshow(dielectric.real, origin='lower')
    plt.colorbar()
    plt.show()

    return wavefunctions_ks, total_energy, density_out


def initial_guess_density(params):
    r"""
    Generate an initial guess for the density: Gaussians centered on atoms scaled by charge
    """

    density = np.zeros(params.Nspace)
    i = 0
    while i < params.num_atoms:

        charge = params.element_charges[params.species[i]]
        density += charge*np.exp(-(params.grid - params.position[i])**2)
        i += 1

    # Normalise
    density *= params.num_particles*(np.sum(density)*params.dx)**-1

    return density


def initial_guess_density_nonint(params):
    r"""
    Initial guess as the solution to the 'non-interacting' problem
    """


    # Independent Hamiltonian
    hamiltonian = construct_hamiltonian_independent(params)

    # Solve H psi = E psi
    eigenvalues, eigenvectors = linalg.eigh(hamiltonian)

    # Extract lowest lying num_particles eigenfunctions and normalise
    wavefunctions_ks = eigenvectors[:,0:params.num_particles]
    wavefunctions_ks[:,0:params.num_particles] = normalise_function(params, wavefunctions_ks[:,0:params.num_particles])

    # Calculate the corresponding output density matrix
    density = calculate_density_ks(params, wavefunctions_ks)

    return density
