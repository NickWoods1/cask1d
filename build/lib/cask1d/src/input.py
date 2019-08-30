import numpy as np

"""
Input to the calculation in the form of a parameters class
"""

class parameters(object):
    def __init__(self,*args,**kwargs):

        # Level of approximation used
        self.method = 'hf'

        # SCF tolerences
        self.tol_ks = 1e-10
        self.tol_hf = 1e-10

        # Size of real space cell
        self.cell = 50
        self.Nspace = 151
        self.dx = self.cell / (self.Nspace - 1)

        # Grid
        self.grid = np.linspace(-0.5*self.cell, 0.5*self.cell, self.Nspace)

        # Stencil order for del^2 operator
        self.stencil = kwargs.pop('stencil',9)

        # Copies of real space cell
        self.supercell = 5

        # List of species + position
        self.species = ['He']
        self.position = [0]

        # SCF
        self.history_length = 10
        self.step_length = 0.01

        # Coulomb softening parameter
        self.soft = 1

        # Have v_ext specified by atoms or given explicitly
        self.manual_v_ext = False

        # Define each element with corresponding atomic number
        self.element_charges = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5,
                                'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10}

        # Number of atoms in the calculation
        self.num_atoms = len(self.species)

        # Number of `electrons'
        num_particles = 0
        for i in range(0,len(self.species)):
            num_particles += self.element_charges[self.species[i]]

        self.num_particles = num_particles
        self.num_electrons = num_particles


