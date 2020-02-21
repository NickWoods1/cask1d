import numpy as np
import matplotlib.pyplot as plt

""" 
Calculate the density response functions
"""


def calculate_susceptibility(params, eigenfunctions, eigenenergies):
    """
    Compute chi(x,x') = dv / dn via first order pertubation theory equation (see, for example, Anglade 2007)
    """

    num_occ = params.num_particles
    susceptibility = np.zeros((params.Nspace, params.Nspace), dtype=complex)

    # Sum over occupied/unoccupied pairs
    for i in range(num_occ):
        for j in range(num_occ,params.Nspace):

            # Difference in energies between occ/unocc states
            delta_energy = eigenenergies[i] - eigenenergies[j]

            f = eigenfunctions[:,i].conj() * eigenfunctions[:,j]
            g = eigenfunctions[:,j].conj() * eigenfunctions[:,i]
            susceptibility[:,:] += 2*np.outer(f,g) / delta_energy

    return susceptibility


def calculate_dielectric(params, density, susceptibility):
    """
    Compute dielectric \epsilon(x,x') = dn_out / dn_in (see Anglade 2007)
    """

    # Compute K_xc(x,x')
    #xc_kernel = -4.437*density**1.61 + 3.381*density**0.61 - 0.7564*density**-0.39
    #xc_kernel = np.zeros(params.Nspace)

    # Compute K_c(x,x') = 1 / |x - x'|
    coulomb_kernel = np.zeros((params.Nspace, params.Nspace))
    for i in range(params.Nspace):
        for j in range(params.Nspace):
            coulomb_kernel[i,j] += 1 / (abs(params.grid[i] - params.grid[j]) + params.soft)

    # Compute residual linear response function, dn_out / dn_in.
    identity = np.eye(params.Nspace)

    dielectric = identity - (coulomb_kernel @ susceptibility)

    return dielectric

