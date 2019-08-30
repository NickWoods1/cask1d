import numpy as np
from cask1d.src.hamiltonian import evaluate_energy_functional, construct_hamiltonian_independent, v_h, v_ext, v_xc
import matplotlib.pyplot as plt

"""
Module containing non-linear root finders to update the density (matrix) in order to find self-consistency, F[n] = n.
"""


def pulay_mixing(params, density_differences, residual_differences, current_residual, current_density_in, i):
    r"""
    As shown in Kresse (1998)
    """

    # Allocates arrays appropriately before and after max history size is reached
    if i >= params.history_length:
        history_size = params.history_length
    elif i < params.history_length:
        history_size = i

    # The Pulay residual dot product matrix
    pulay_matrix = np.zeros((history_size,history_size))

    # The RHS vector for the Pulay linear system Ax=b
    b = np.zeros(history_size)
    for j in range(0,history_size):
        b[j] = -np.dot(residual_differences[j,:], current_residual[:])

    # Construct Pulay matrix
    k = 0
    for j in range(0,history_size):
        for k in range(0,history_size):

                # Pulay matrix is matrix of dot products of residuals
                pulay_matrix[j,k] = np.dot(residual_differences[j],residual_differences[k])
                pulay_matrix[k,j] = pulay_matrix[j,k]

                k += 1

    # Solve for the (Pulay) optimal coefficients
    pulay_coefficients = np.linalg.solve(pulay_matrix, b)

    # Final Pulay update: n_new = n_opt + \alpha R_opt
    density_in = current_density_in + params.step_length*current_residual
    for j in range(0,history_size):
        density_in[:] += pulay_coefficients[j]*(density_differences[j,:] + params.step_length*residual_differences[j,:])

    # Pulay predicing negative densities?!
    density_in = abs(density_in)

    return density_in


def ODA(params, dmatrix_in, dmatrix_out):
    r"""
    Optimal damping algorithm (Cances 2001)
    """

    dmatrix_ref = np.copy(dmatrix_in)
    search_direction = dmatrix_out - dmatrix_in

    x, y = np.zeros(3), np.zeros(3)

    # Energy with no step
    energy0 = ODA_energy_functional(params, dmatrix_ref)
    x[0] = 0
    y[0] = energy0

    # Energy first step
    alpha = 0.4
    energy1 = ODA_energy_functional(params, dmatrix_ref + alpha*search_direction)
    x[1] = alpha
    y[1] = energy1

    # Energy second step
    alpha = 0.8
    energy2 = ODA_energy_functional(params, dmatrix_ref + alpha*search_direction)
    x[2] = alpha
    y[2] = energy2

    # Fit quadratic
    coeffs = np.polynomial.polynomial.polyfit(x, y, 2)

    optimal_step = coeffs[1] / (2*coeffs[2])

    if optimal_step > 1 or optimal_step < 0:
        optimal_step = 1

    return optimal_step


def ODA_energy_functional(params, dmatrix):
    r"""
    Evaluates the KS energy functional E[n] for a given density + orbitals
    """

    density = np.diagonal(dmatrix)

    hamiltonian_indep = construct_hamiltonian_independent(params)
    product = np.dot(hamiltonian_indep,dmatrix)
    indep_energy = 0
    for i in range(0,params.Nspace):
        indep_energy += product[i,i]*params.dx

    # Hartree energy
    hartree_pot = v_h(params, density)
    hartree_energy = np.sum(density*hartree_pot)*params.dx

    # Exchange energy
    exchange_energy = 0
    for i in range(0,params.Nspace):
        for j in range(0,params.Nspace):
            exchange_energy -= dmatrix[i,j]**2 / (abs(params.grid[i] - params.grid[j]) + params.soft)

    exchange_energy *= params.dx**2
    total_energy = indep_energy + hartree_energy + exchange_energy

    return total_energy
