import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from density2potential.utils.physics import element_charges, calculate_density_ks
from density2potential.utils.math import discrete_Laplace, normalise_function
import scipy.linalg as linalg
from cask1d.src.hamiltonian import construct_hamiltonian, v_h, v_ext, v_xc
from cask1d.src.scf import pulay_mixing

"""
Computes the self-consistent Kohn-Sham orbitals, density, and energy given a density functional and external
potential 
"""


def minimise_energy_dft(params):

    # Array that will store SCF iterative densities and residuals
    history_of_densities_in = np.zeros((params.history_length, params.Nspace))
    history_of_densities_out = np.zeros((params.history_length, params.Nspace))
    history_of_residuals = np.zeros((params.history_length, params.Nspace))
    density_differences = np.zeros((params.history_length, params.Nspace))
    residual_differences = np.zeros((params.history_length, params.Nspace))

    # Generate initial guess density (sum weighted Gaussians)
    density_in = initial_guess_density(params)

    # SCF loop
    i, error = 0, 1
    while error > 1e-10:

        # Iteration number modulus history length
        i_mod = i % params.history_length
        i_mod_prev = (i-1) % params.history_length

        # Construct Hamiltonian
        hamiltonian = construct_hamiltonian(params, density_in)

        # Solve H psi = E psi
        eigenvalues, eigenvectors = linalg.eigh(hamiltonian)

        # Extract lowest lying num_particles eigenfunctions and normalise
        wavefunctions_ks = eigenvectors[:,0:params.num_particles]
        wavefunctions_ks[:,0:params.num_particles] = normalise_function(params, wavefunctions_ks[:,0:params.num_particles])

        # Calculate the output density
        density_out = calculate_density_ks(params, wavefunctions_ks)

        # Calculate total energy
        total_energy = calculate_total_energy(params, eigenvalues[0:params.num_particles], density_in)
        print(evaluate_ks_functional(params, wavefunctions_ks, density_out))


        # L1 error between input and output densities
        error = np.sum(abs(density_in - density_out)*params.dx)
        print('SCF error = {0} at iteration {1} with energy {2}'.format(error, i, total_energy))

        # Store densities/residuals within the iterative history data
        history_of_densities_in[i_mod,:] = density_in
        history_of_densities_out[i_mod,:] = density_out
        history_of_residuals[i_mod,:] = density_out - density_in

        if i == 0:

            # Damped linear step for the first iteration
            density_in = density_in - params.step_length * (density_in - density_out)

        elif i > 0:

            # Store more iterative history data...
            density_differences[i_mod_prev] = history_of_densities_in[i_mod] - history_of_densities_in[i_mod_prev]
            residual_differences[i_mod_prev] = history_of_residuals[i_mod] - history_of_residuals[i_mod_prev]

            # Perform Pulay step using the iterative history data
            density_in = pulay_mixing(params, density_differences, residual_differences,
                                      history_of_residuals[i_mod], history_of_densities_in[i_mod], i)

        i += 1

    plt.plot(density_out)
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
    density *= params.num_atoms*(np.sum(density)*params.dx)**-1

    return density


def evaluate_ks_functional(params, wavefunctions_ks, density):
    r"""
    Evaluates the KS energy functional E[n] for a given density + orbitals
    """

    # Kinetic energy
    kinetic_energy = 0
    laplace = discrete_Laplace(params)
    for i in range(0,params.num_particles):
        del_sq_phi = np.dot(laplace,wavefunctions_ks[:,i])
        kinetic_energy += np.sum(np.conj(wavefunctions_ks[:,i])*del_sq_phi)

    kinetic_energy *= -0.5*params.dx

    # Hartree energy
    hartree_pot = v_h(params, density)
    hartree_energy = 0.5*np.sum(density*hartree_pot)*params.dx

    # External
    external_pot = v_ext(params)
    external_energy = np.sum(density*external_pot)*params.dx

    # XC energy
    xc_pot = v_xc(density)
    xc_energy = np.sum(density*xc_pot)*params.dx

    total_energy = kinetic_energy + hartree_energy + external_energy + xc_energy

    return total_energy

#def evaluate_hf_functional():

#def evaluate_h_functional():


def calculate_total_energy(params, eigenenergies, density):
    r"""
    Calculates the total energy given a set of occupied eigenenergies
    and corresponding density
    """

    # Hartree energy
    # Add Hartree potential
    v_h = np.zeros(params.Nspace)
    for i in range(0,params.Nspace):
        for j in range(0,params.Nspace):
            v_h[i] += density[j] / (abs(params.grid[i] - params.grid[j]) + params.soft)
    v_h *= params.dx
    E_h = 0.5*np.sum(density * v_h)*params.dx

    total_energy = np.sum(eigenenergies) - E_h

    return total_energy
