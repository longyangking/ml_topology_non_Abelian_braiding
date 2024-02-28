import numpy as np
import itertools
from .non_Hermitian_Hamiltonian import BraidOperator
from .generator import obtain_model_braid

class UnknotGenerator:
    def __init__(self, n_band):
        self.n_band = n_band

    def generate(self):
        models = list()
        braid_operators_index_list = list(itertools.permutations(range(1, self.n_band), self.n_band-1))

        for status in [False, True]:
            for i, braid_operator_indices in enumerate(braid_operators_index_list):
                braid_operators = [BraidOperator(index, status) for index in braid_operator_indices]
                models.append(obtain_model_braid(n_band=self.n_band, braid_operators=braid_operators))

        return models