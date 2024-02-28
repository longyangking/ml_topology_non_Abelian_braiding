import numpy as np
import numpy.linalg as LA

class BraidOperator:
    def __init__(self, n, is_inverse=False):
        self.n = n # start from 
        self.is_inverse = is_inverse

        if (self.n < 0) and (self.is_inverse):
            self.n = - self.n
            self.is_inverse = False

        elif (self.n < 0) and (not self.is_inverse):
            self.n = - self.n
            self.is_inverse = True

    def __ne__(self, operator) -> bool:
        if operator.get_n() != self.get_n():
            return True
        
        if operator.get_is_inverse() != self.get_is_inverse():
            return True
        
        return False
    
    def __eq__(self, operator) -> bool:
        return not self.__ne__(operator)
 
    def get_n(self):
        return self.n
    
    def get_is_inverse(self):
        return self.is_inverse
    
    def __str__(self):
        return self.__expr__()

    def __expr__(self):
        info = r"$\sigma_{n}$".format(n=self.n) if not self.is_inverse else r"$\sigma^{{-1}}_{n}$".format(n=self.n)
        return info
    
class NonHermitianHamiltonianBraid:
    def __init__(self, n_band, braid_operators):
        self.n_band = n_band
        self.braid_operators = braid_operators
        self.n_operator = len(self.braid_operators)

        self.Ns = None
        self.initiate_effective()

    def get_braid_words(self):
        return self.braid_operators
    
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

    def get_n_band(self):
        return self.n_band

    def __call__(self, k):
        return self.get_Hamiltonian(k)
    
    def get_Hamiltonian(self, k):
        return np.diag(self.get_energies(k))
    
    def get_energies(self, k):
        #k = k%b # contraint it in the first BZ

        bands = np.array([i for i in range(1, self.n_band+1)], dtype=complex)
        band_indices = np.array([i for i in range(self.n_band)])

        if self.n_operator == 0:
            return bands

        index_step = 0
        k_step = 2*np.pi/self.n_operator
        while (index_step + 1)*k_step <= k:
            index_step = index_step + 1
        
        if index_step > 0:
            for i_step in range(index_step):
                index = i_step % self.n_operator
                braid_operator = self.braid_operators[index]
                n = braid_operator.get_n()
                m = n - 1 # the label of the operator starts from 1, but the index of the band starts from 0
                is_inverse = braid_operator.get_is_inverse()

                En0 = bands[band_indices[m]]
                Enp0 = bands[band_indices[m+1]]
                Ec = (En0 + Enp0)/2.
                theta = np.pi
                sign = -1 if not is_inverse else 1

                En = Ec + (En0 - Ec)*np.exp(1j*theta*sign)
                Enp = Ec + (Enp0 - Ec)*np.exp(1j*theta*sign)
                
                bands[band_indices[m]] = En
                bands[band_indices[m+1]] = Enp

                band_indices = np.argsort(np.real(bands))

        index = index_step % self.n_operator
        braid_operator = self.braid_operators[index]
        n = braid_operator.get_n()
        m = n - 1 # the label of the operator starts from 1, but the index of the band starts from 0
        is_inverse = braid_operator.get_is_inverse()

        En0 = bands[band_indices[m]]
        Enp0 = bands[band_indices[m+1]]
        Ec = (En0 + Enp0)/2.
        theta = self.n_operator/2*(k - index_step*2*np.pi/self.n_operator) # = \pi/(2\pi/n_operator)*k
        sign = -1 if not is_inverse else 1

        En = Ec + (En0 - Ec)*np.exp(1j*theta*sign)
        Enp = Ec + (Enp0 - Ec)*np.exp(1j*theta*sign)
        
        bands[band_indices[m]] = En
        bands[band_indices[m+1]] = Enp

        return bands
    
    def get_subspace_Hamiltonian(self, k):
        bands = self.get_energies(k)
        return np.array([[bands[j]-bands[i] for j in range(self.n_band)] for i in range(self.n_band)])

    def initiate_effective(self, eps_k=0.001*np.pi):
        # eps_k is for the numerical stability
        dk = 2*np.pi/self.n_operator if self.n_operator > 0 else 2*np.pi
        _coeff = np.zeros((self.n_band, self.n_band), dtype=complex)

        for ip in range(self.n_operator):
            _sub_hk_start = self.get_subspace_Hamiltonian(dk*ip)
            _sub_hk_end = self.get_subspace_Hamiltonian(dk*(ip+1)-eps_k) 

            for i in range(self.n_band):
                for j in range(i+1, self.n_band):
                    _coeff[i,j] = _coeff[i,j] + np.angle(_sub_hk_end[i,j]/_sub_hk_start[i,j])
                    _coeff[j,i] = _coeff[i,j]

        self.Ns = _coeff/(2*np.pi)
        self.Ns = np.round(2*self.Ns)/2
        self.Ns = np.real(self.Ns)
        
        return self.Ns #, _coeff

    def get_effective_subspace_Hamiltonian(self, k):
        return np.array([[np.exp(1j*self.Ns[i,j]*k) for j in range(self.n_band)] for i in range(self.n_band)])

class NonHermitianHamiltonianTorusKnot:
    def __init__(self, q, p):
        '''
        Torus knot (n,m)
        '''
        self.p = int(p)
        self.q = int(q)

    def __call__(self, k):
        return self.get_Hamiltonian(k)
    
    def get_n_band(self):
        return self.q
    
    def get_Hamiltonian(self, k):
        return np.diag(self.get_energies(k))

    def get_energies(self, k, e0=1.):
        # phis = [np.angle(np.exp(1j*2*np.pi/self.m*v)) for v in range(self.m)]
        bands = [e0*np.exp(1j*(self.p*k/self.q + 2*np.pi/self.q*v)) for v in range(self.q)]
        return bands
    
    def get_subspace_Hamiltonian(self, k):
        bands = self.get_energies(k)
        return np.array([[bands[j]-bands[i] for j in range(self.n_band)] for i in range(self.n_band)])

    def initiate_effective(self):
        pass
    
    def get_effective_subspace_Hamiltonian(self, k):
        return self.get_subspace_Hamiltonian(k)
        #return np.array([[np.exp(1j*self.Ns[i,j]*k) for j in range(self.n_band)] for i in range(self.n_band)])
