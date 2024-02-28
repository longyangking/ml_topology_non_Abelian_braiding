'''
Generate the perturbations
'''

import numpy as np

sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,-1j],[1j,0]])
sigma_z = np.array([[1,0],[0,-1]])
sigma_0 = np.array([[1,0],[0,1]]) 


class PerturbationModel:
    def __init__(self):
        self.t1 = None
        self.t2 = None
        self.initiate()

    def initiate(self):
        self.t1 = (-1+2*np.random.random())*0.1
        self.t2 = (-1+2*np.random.random())*0.1

    def get_parameters(self):
        return self.t1, self.t2
    
    def set_parameters(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def get_Hamiltonian(self, k):
        return self.t1*sigma_x + self.t2*sigma_y