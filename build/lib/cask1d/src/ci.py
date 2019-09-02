import numpy as np
from cask1d.src.hf import calculate_dmatrix
from itertools import permutations# import itertools
#from sympy.combinatorics import Permutation



"Configuration Interaction from some reference orbitals"


def solve_ci_groundstate(params):#, wavefunctions):

    #dmatrix = calculate_dmatrix(params, wavefunctions)
    #density = np.diagonal(dmatrix)

    #perms = itertools.permutations([1,2,3,4])

    #for p in perms:
    #    l = list(p)
    #    print(perm_parity(l), p)

    #perms = Permutation([0,1,2,3,4])

    #print(perms)

    for perms in permutations(range(4)):
        l = list(perms)
        print(l)
        #print(perm_parity(l), perms)

        for i in l:
            print(i)


        #wavefunction += perm_parity(p) * ()



def perm_parity(lst):
    '''\
    Given a permutation of the digits 0..N in order as a list,
    returns its parity (or sign): +1 for even parity; -1 for odd.
    '''
    parity = 1
    for i in range(0,len(lst)-1):
        if lst[i] != i:
            parity *= -1
            mn = min(range(i,len(lst)), key=lst.__getitem__)
            lst[i],lst[mn] = lst[mn],lst[i]
    return parity
