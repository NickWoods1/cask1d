import numpy as np

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
