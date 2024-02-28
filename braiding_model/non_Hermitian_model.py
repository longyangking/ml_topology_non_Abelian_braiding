import numpy as np
import numpy.linalg as LA

import numba as nb
from numba.typed import List

@nb.jit()
def get_P(P, allvectors, n):
    for index in range(len(allvectors)):  # for the each k point
        for i in range(n):
            for j in range(n):
                for vector in allvectors[index]:  # for the each band
                        P[index, i, j] += np.conjugate(vector[j])*vector[i]

class SubspaceModel:
    def __init__(self, n_dim, n, index, Hamiltonian, is_effective=False):
        self.n_dim = n_dim # the dimension
        self.n = n # the number of bands
        self.index = index
        self.Hamiltonian = Hamiltonian
        self.is_effective = is_effective

    def get_n_dim(self):
        return self.n_dim
    
    def get_n(self):
        return self.n

    def get_n_band(self):
        return self.Hamiltonian.get_n_band()

    def get_Hamiltonian(self, k):
        if self.is_effective:
            hk = self.Hamiltonian.get_effective_subspace_Hamiltonian(k)
        else:
            hk = self.Hamiltonian.get_subspace_Hamiltonian(k)         

        hk = hk[self.index]
        hk = np.array([[0,hk],[np.conjugate(hk),0]],dtype=complex)
        return hk
    
    def calculate_Q(self, kpoints, perturbation=None):
        P = self.calculate_projection_operator(kpoints=kpoints, perturbation=perturbation)
        n_P = len(P)
        Q = np.zeros((n_P, self.n, self.n), dtype=complex)
        for n in range(n_P):
            Q[n] = np.identity(self.n) - 2*P[n]

        return Q

    def calculate_projection_operator(self, kpoints, perturbation=None):
        # calculate the projection operator
        # if self.n_dim == 0:
        #     values, vectors = self.get_eigensystem(0, perturbation=perturbation)
        #     index = np.where(values<0.0)
        #     vectors = vectors[index]

        #     P = np.zeros((self.n, self.n), dtype=complex)
        #     for i in range(self.n):
        #         for j in range(self.n):
        #             for vector in vectors:
        #                 P[i,j] += np.conjugate(vector[j])*vector[i]

        #     return [P]
        
        allvectors = list()

        # kpoints = self.get_kpoints()
        for k in kpoints:
            values, vectors = self.get_eigensystem(k, perturbation=perturbation)
            index = np.where(values<0.0)
            vectors = vectors[index]
            allvectors.append(vectors[index])

        P = np.zeros((len(allvectors), self.n, self.n), dtype=complex)
        typed_allvectors = List()
        [typed_allvectors.append(x) for x in allvectors]
        get_P(P, typed_allvectors, self.n)

        # values, vectors = self.get_eigensystem(k, perturbation=perturbation)
        # index = np.where(values<0.0)
        # vectors = vectors[index]

        # P = np.zeros((self.n, self.n), dtype=complex)
        # for i in range(self.n):
        #     for j in range(self.n):
        #         for vector in vectors:
        #             P[i,j] += np.conjugate(vector[j])*vector[i]

        return P

    def get_eigensystem(self, k, perturbation=None):
        hk = self.get_Hamiltonian(k)

        hp = 0
        if perturbation is not None:
            hp = perturbation.get_Hamiltonian(k)
        hk = hk + hp

        if np.allclose(hk, np.transpose(np.conjugate(hk))): # Hermitian matrix
            values, vectors = LA.eigh(hk)
        else:
            raise Exception("Resulted Hamiltonian after Hermitian flattening is Non-Hermitian !")
        
        vectors = np.transpose(vectors)
        
        index = np.argsort(values)
        values = values[index]
        vectors = vectors[index]
        return values, vectors
   
class NonHermitianModel:
    def __init__(self, Hamiltonian, verbose=False):
        self.Hamiltonian = Hamiltonian
        self.verbose = verbose
        self.n_band = self.Hamiltonian.get_n_band()
        self.__winding_matrix = None

    def get_winding_matrix(self):
        if self.__winding_matrix is None:
            self.__winding_matrix = self.Hamiltonian.initiate_effective()
        return self.__winding_matrix

    def get_n_band(self):
        return self.n_band
    
    def get_braid_words(self):
        return self.Hamiltonian.get_braid_words()
    
    def __ne__(self, model_compare) -> bool:
        braid_words = self.get_braid_words()
        braid_words_compare = model_compare.get_braid_words()

        n_braid_words = len(braid_words)
        n_braid_words_compare = len(braid_words_compare)

        if n_braid_words != n_braid_words_compare:
            return True
        
        for n in range(n_braid_words):
            if braid_words[n] != braid_words_compare[n]:
                return True
            
        return False
    
    def __eq__(self, model_compare) -> bool:
        return not self.__ne__(model_compare)

    def get_energies(self, k):
        energies = self.Hamiltonian.get_energies(k)
        return energies

    def get_subspace_Hamiltonian(self, k, is_effective=True):
        if is_effective:
            return self.Hamiltonian.get_effective_subspace_Hamiltonian(k)
        
        return self.Hamiltonian.get_subspace_Hamiltonian(k)
    
    def get_subspace_models(self, is_effective=True):
        subspace_models = list()
        for i in range(self.n_band):
            for j in range(i+1, self.n_band): # only half to make subspace
                subspace_models.append(
                    SubspaceModel(n_dim=1, n=2, index=(i,j), Hamiltonian=self.Hamiltonian, is_effective=is_effective)
                )
        
        return subspace_models
    
    def save_band(self, filename, n_k=101):
        ks = np.linspace(0,2*np.pi, n_k)
        Es = np.zeros((n_k, 2*self.n_band))
        for i, k in enumerate(ks):
            _E = self.get_energies(k)
            for n in range(self.n_band):
                Es[i, 2*n] = np.real(_E[n])
                Es[i, 2*n+1] = np.imag(_E[n])

        np.savetxt(filename, Es)
    # def get_eigensystem_Hamiltonian_subspace(self, k, perturbation=None):
        
    #     hp = 0
    #     if hp is not None:
    #         hp = perturbation.get_Hamiltonian(k)

    #     original_hamiltonians = self.get_subspace_Hamiltonian(k)
    #     hamiltonians = [[
    #         np.array([[0, original_hamiltonians[i,j]],[np.conjugate(original_hamiltonians[i,j]), 0]]) + hp
    #         for j in range(self.n_band)] for i in range(self.n_band)]
    
def get_subspace_model(non_Hermitian_Hamiltonian):
    '''
    A N-band non-Hermitian model should return N*N subspace models under chiral symmetry, 
    but in which N subspace models are trivial due to all zero elements.
    '''
    pass
