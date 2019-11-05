import numpy as np
import matplotlib.pyplot as plt

""" 
Calculate the density response functions
"""


def calculate_susceptibility_idea(params, eigenfunctions, eigenenergies):
    """
    Compute chi(x,x') = dv / dn via first order pertubation theory equation (see, for example, Anglade 2007)
    """

    num_occ = params.num_particles
    susceptibility = np.zeros((params.Nspace, params.Nspace), dtype=complex)

    l = 0
    for i in range(num_occ):
        for j in range(num_occ,params.Nspace):
            l += 1

    amplitudes = np.zeros((params.Nspace, params.Nspace, l))
    exct_energy = np.zeros(l)

    l = 0
    # Sum over occupied/unoccupied pairs
    for i in range(num_occ):
        for j in range(num_occ,params.Nspace):

            psi_occ = eigenfunctions[:,i]
            psi_unocc = eigenfunctions[:,j]
            product = np.multiply(psi_occ,psi_unocc)

            for k in range(params.Nspace):
                amplitudes[k,:,l] = product[k]*product[:]

            exct_energy[i] = eigenenergies[j] - eigenenergies[i]

            l += 1

    for i in range(num_occ):
        for j in range(num_occ,params.Nspace):

            ee = exct_energy[i]

            susceptibility[:,:] += -2.0*amplitudes[:,:,k] / ee

    return susceptibility


def calculate_susceptibility(params, eigenfunctions, eigenenergies):
    """
    Compute chi(x,x') = dv / dn via first order pertubation theory equation (see, for example, Anglade 2007)
    """

    num_occ = params.num_particles
    susceptibility = np.zeros((params.Nspace, params.Nspace), dtype=complex)

    # Sum over occupied/unoccupied pairs
    for i in range(num_occ): #n
        for j in range(num_occ,params.Nspace):#n'

            # Difference in energies between occ/unocc states
            delta_energy = eigenenergies[i] - eigenenergies[j]

            for k in range(params.Nspace): #r
                for l in range(params.Nspace):#r'

                    susceptibility[k,l] += 2*eigenfunctions[k,i].conj()*eigenfunctions[k,j]*eigenfunctions[l,j].conj()*eigenfunctions[l,i] / delta_energy

            #f = eigenfunctions[:,i].conj() * eigenfunctions[:,j]
            #g = eigenfunctions[:,j].conj() * eigenfunctions[:,i]
            #susceptibility[:,:] += 2*np.outer(f,g) / delta_energy

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

