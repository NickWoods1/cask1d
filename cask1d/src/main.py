import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
from cask1d.src.input import parameters
from cask1d.src.dft import minimise_energy_dft
from cask1d.src.hf import minimise_energy_hf
from cask1d.src.tf import minimise_energy_tf
from cask1d.src.ci import solve_ci_groundstate
from cask1d.src.hamiltonian import v_ext

"""
Entry point for the requested action
"""

def main():

    __version__ = 0.1

    # Parser class for density2potential
    parser = argparse.ArgumentParser(
        prog='cask1d',
        description='Find the ground state energy of  '
                    ' a molecular system given an approximation',
        epilog='written by Nick Woods')

    # Specify arguments that the package can take
    parser.add_argument('--version', action='version', version='This is version {0} of cask1d.'.format(__version__))
    parser.add_argument('task', help='what do you want cask1d to do: dft, hf, h')

    args = parser.parse_args()

    # Code header
    print('    ------------------------')
    print('             cask1d         ')
    print('    ------------------------')
    print('           Written by')
    print('           Nick Woods')
    print(' ')

    # Find the Kohn-Sham potential that generates a given reference density
    if args.task == 'get-groundstate':

        # Construct parameters class
        params = parameters()

        if params.method == 'hf':
            wavefunctions, total_energy, density = minimise_energy_hf(params)
        elif params.method == 'dft':
            wavefunctions, total_energy, density = minimise_energy_dft(params)
        elif params.method == 'h':
            # DFT without the density functional
            wavefunctions, total_energy, density = minimise_energy_dft(params)
        elif params.method == 'tf':
            density, energy = minimise_energy_tf(params)

        # Save output wavefunctions and density
        if params.method != 'tf':
            np.save('wavefunctions.npy',wavefunctions)

        np.save('density_{}.npy'.format(params.method),density)

        # Plot output
        plt.plot(density,label='Ground state {} density'.format(params.method))
        plt.legend()
        plt.savefig('gs_density_{}.pdf'.format(params.method))

        plt.clf()
        plt.plot(v_ext(params),label='{} external potential'.format(params.method))
        plt.legend()
        plt.savefig('{}_external_potential.pdf'.format(params.method))

    if args.task == 'plot':

        # Given the output of a HF, H, and DFT calculation, plot three output densities.
        hf = np.load('density_hf.npy')
        h = np.load('density_h.npy')
        dft = np.load('density_dft.npy')

        plt.plot(hf,label='hf')
        plt.plot(h,label='h')
        plt.plot(dft,label='dft')
        plt.legend()
        plt.savefig('compare.pdf')

