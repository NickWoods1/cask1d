import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
from cask1d.src.input import parameters
from cask1d.src.dft import minimise_energy_dft
from cask1d.src.hf import minimise_energy_hf

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
    if args.task == 'dft':

        # Construct parameters class
        params = parameters()

        #minimise_energy_hf(params)
        minimise_energy_dft(params)




