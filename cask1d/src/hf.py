import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from density2potential.utils.physics import calculate_density_ks
from density2potential.utils.math import discrete_Laplace, normalise_function
import scipy.linalg as linalg
from cask1d.src.hamiltonian import construct_hamiltonian_independent, update_hamiltonian, evaluate_energy_functional
from cask1d.src.hamiltonian import v_h, fock_exchange, v_ext
from cask1d.src.scf import pulay_mixing, ODA

"""
Computes the self-consistent Hartree-Fock orbitals, density, and energy given an external
potential 
"""


def minimise_energy_hf(params):
    r"""
    Compute ground state Hartree-Fock solution
    """

    # Generate initial guess density (sum weighted Gaussians)
    dmatrix_in = initial_guess_dmatrix(params)
    density_in = np.diagonal(dmatrix_in)

    # Construct the independent part of the hamiltonian, i.e. KE + v_external
    hamiltonian_independent = construct_hamiltonian_independent(params)

    # SCF loop
    i, error = 0, 1
    while error > params.tol_hf:

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

        step_length = ODA(params, dmatrix_in, dmatrix_out)
        dmatrix_in = dmatrix_in - step_length * (dmatrix_in - dmatrix_out)
        density_in = np.diagonal(dmatrix_in)

        i += 1

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
        for n in range(0,params.Nspace):
            for m in range(0,params.Nspace):
                dmatrix[n,m] += np.conj(wavefunctions_ks[n,i])*wavefunctions_ks[m,i]

    return dmatrix
