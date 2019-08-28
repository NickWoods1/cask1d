import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from density2potential.utils.physics import element_charges, calculate_density_ks
from density2potential.utils.math import discrete_Laplace, normalise_function
import scipy.linalg as linalg
from cask1d.src.hamiltonian import construct_hamiltonian_independent, update_hamiltonian, evaluate_energy_functional
from cask1d.src.scf import pulay_mixing

"""
Computes the self-consistent Hartree-Fock orbitals, density, and energy given an external
potential 
"""


def minimise_energy_hf(params):

    # Array that will store SCF iterative densities and residuals
    #history_of_densities_in = np.zeros((params.history_length, params.Nspace, params.Nspace))
    #history_of_densities_out = np.zeros((params.history_length, params.Nspace, params.Nspace))
    #history_of_residuals = np.zeros((params.history_length, params.Nspace, params.Nspace))
    #density_differences = np.zeros((params.history_length, params.Nspace, params.Nspace))
    #residual_differences = np.zeros((params.history_length, params.Nspace, params.Nspace))

    # Generate initial guess density (sum weighted Gaussians)
    dmatrix_in = initial_guess_dmatrix(params)
    density_in = np.diagonal(dmatrix_in)

    plt.plot(density_in)
    plt.show()

    # Construct the independent part of the hamiltonian, i.e. KE + v_external
    hamiltonian_independent = construct_hamiltonian_independent(params)

    # SCF loop
    i, error = 0, 1
    while error > 1e-10:

        # Update hamiltonian with the orbital-dependent terms
        hamiltonian = update_hamiltonian(params, hamiltonian_independent, density_in, dmatrix_in)

        # Solve H psi = E psi
        eigenvalues, eigenvectors = linalg.eigh(hamiltonian)

        # Extract lowest lying num_particles eigenfunctions and normalise
        wavefunctions_ks = eigenvectors[:,0:params.num_particles]
        wavefunctions_ks[:,0:params.num_particles] = normalise_function(params, wavefunctions_ks[:,0:params.num_particles])

        # Calculate the output density
        dmatrix_out = calculate_dmatrix_ks(params, wavefunctions_ks)
        density_out = np.diagonal(dmatrix_out)

        # Calculate total energy
        total_energy = evaluate_energy_functional(params, wavefunctions_ks, density_out, dmatrix_out)

        # L1 error between input and output densities
        error = np.linalg.norm(dmatrix_in - dmatrix_out)
        print('SCF error = {0} at iteration {1} with energy {2}'.format(error, i, total_energy))

        # Damped linear step for the first iteration
        dmatrix_in = dmatrix_in - params.step_length * (dmatrix_in - dmatrix_out)

        i += 1

    np.save('hf_density.npy',density_out)
    plt.plot(density_out)
    plt.show()

    return wavefunctions_ks, total_energy, density_out


def initial_guess_dmatrix(params):
    r"""
    Initial guess density matrix by running a calculation with just the independent parts of the Hamiltonian
    """

    # Independent Hamiltonian
    hamiltonian = construct_hamiltonian_independent(params)

    # Solve H psi = E psi
    eigenvalues, eigenvectors = linalg.eigh(hamiltonian)

    # Extract lowest lying num_particles eigenfunctions and normalise
    wavefunctions_ks = eigenvectors[:,0:params.num_particles]
    wavefunctions_ks[:,0:params.num_particles] = normalise_function(params, wavefunctions_ks[:,0:params.num_particles])

    # Calculate the corresponding output density matrix
    dmatrix = calculate_dmatrix_ks(params, wavefunctions_ks)

    return dmatrix


def calculate_dmatrix_ks(params, wavefunctions_ks):
    r"""
    Computes the density matrix corresponding to a set of orbitals
    """

    dmatrix = np.zeros((params.Nspace,params.Nspace))

    for i in range(0,params.num_particles):
        dmatrix += np.conj(wavefunctions_ks[:,i])*np.conj(wavefunctions_ks[:,i])

    return dmatrix
