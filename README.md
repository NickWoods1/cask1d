*********
## CASK1d
*********

`cask1d` is a python package designed to compute the ground
state energy and density of a molecular system in one-dimension using
various approximations. `cask1d` is non-periodic
 and uses a delta function basis set. 

*****************
### Functionality
*****************

Given an external potential, specified manually (e.g. harmonic well), or using the in-built utility
to specify atomic species & positions:

`dft` generates the Kohn-Sham self-consistent single-particle
wavefunctions, total energy, and density, given an exchange-correlation potential (currently, LDA).

`h` generates Hartree self-consistent single-particle 
wavefunctions, total energy, and density.

`hf` generates Hartree-Fock self-consistent single-particle
wavefunctions, total energy, and density. 

`tf` generates Thomas-Fermi density and energy (Dirac exchange term optional)

********************
### Work in progress
********************

##### Partially complete:
 
 Calculation of linear response functions -- analysis and density mixing. 

##### Just begun:
 
 `ci` Configuration interaction

##### Plan:
 
 `cc` Coupled cluster
 
 `gw` GW approximation from given a reference
 