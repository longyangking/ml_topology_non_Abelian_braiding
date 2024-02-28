import numpy as np
from itertools import permutations, product
from .non_Hermitian_model import NonHermitianModel
from .non_Hermitian_Hamiltonian import NonHermitianHamiltonianBraid, NonHermitianHamiltonianTorusKnot

def obtain_model_braid(n_band, braid_operators):
    hamiltonian = NonHermitianHamiltonianBraid(n_band=n_band, braid_operators=braid_operators)
    model = NonHermitianModel(hamiltonian)
    return model

def obtain_model_torus_knot(p, q):
    hamiltonian = NonHermitianHamiltonianTorusKnot(q=q, p=p)
    model = NonHermitianModel(hamiltonian)
    return model

class GeneratorBraid:
    def __init__(self, n_band, braid_operator_list, verbose=False):
        self.n_band = n_band
        self.braid_operator_list = braid_operator_list
        self.__check()
        self.verbose = verbose

    def __check(self):
        for braid_operator in self.braid_operator_list:
            if braid_operator.get_n() >= self.n_band:
                raise Exception("Error: the n of the braid operator must be smaller than the number of the bands")

    def get_n_band(self):
        return self.n_band
    
    def get_braid_operator_list(self):
        return self.braid_operator_list
    
    def get_random_generate_sample(self, n_sample, n_max_operator=5):
        '''
        n_max_operators contrains the maximum number of braid operators
        '''
        models = list()
        braid_operator_list_gen = list() 
        n_operator_list = len(self.braid_operator_list)

        for _ in range(n_sample):
            n_operator = 1 + np.random.randint(n_max_operator)
            indices = np.random.randint(n_operator_list, size=n_operator)
            braid_operators = [self.braid_operator_list[index] for index in indices]

            models.append(obtain_model_braid(
                n_band=self.n_band,
                braid_operators=braid_operators
                ))
            braid_operator_list_gen.append(braid_operators)

        return models, braid_operator_list_gen
    
    def get_permutation_generate_sample(self, n_permutation):
        '''
        combination of the operators
        '''
        models = list()
        braid_operator_list_gen = list() 
        n_operator_list = len(self.braid_operator_list)

        # if n_permutation > n_operator_list:
        #     n_permutation = n_operator_list

        perms = permutations(range(n_operator_list), n_permutation)

        for perm in perms:
            braid_operators = [self.braid_operator_list[index] for index in perm]
            models.append(obtain_model_braid(
                n_band=self.n_band,
                braid_operators=braid_operators
                ))
            braid_operator_list_gen.append(braid_operators)

        return models, braid_operator_list_gen
    
    def get_all_combination_generate_sample(self, n_length):
        '''
        (n_operatpr)**n_length
        '''
        models = list()
        braid_operator_list_gen = list() 
        n_operator_list = len(self.braid_operator_list)

        combination_list = list(product(*[range(n_operator_list) for _ in range(n_length)]))

        #n_sample = n_operator_list**n_length
        for _combination in combination_list:
            braid_operators = [self.braid_operator_list[index] for index in _combination]

            models.append(obtain_model_braid(
                n_band=self.n_band,
                braid_operators=braid_operators
                ))
            braid_operator_list_gen.append(braid_operators)

        return models, braid_operator_list_gen
    
