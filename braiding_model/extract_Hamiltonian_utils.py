import numpy as np
from .non_Hermitian_Hamiltonian import NonHermitianHamiltonianBraid
from .non_Hermitian_model import NonHermitianModel
import numpy.linalg as LA

class NonHamiltonianExtract(NonHermitianHamiltonianBraid):
    def __init__(self, Hamiltonian_func, n_band, compress_ratio=0.75, axis=1.):
        self.Hamiltonian_func = Hamiltonian_func

        self.compress_ratio = compress_ratio
        self.axis = axis/np.abs(axis)
        
        super().__init__(n_band=n_band, braid_operators=[])
        self.__check_axis()

    def __check_axis(self, epsilon=1e-3):
        flag = False
        while not flag:
            flag = True
            bands = self.get_energies(0)
            for i in range(self.n_band):
                for j in range(i+1, self.n_band):
                    if np.abs(bands[i]-bands[j]) < epsilon:
                        flag = False
                        break
                
                if not flag:
                    break
            
            if not flag:
                theta = np.angle(self.axis)
                theta = theta + 0.01*np.pi # constant step
                self.axis = np.exp(1j*theta)                  

    def get_axis(self):
        return self.axis

    def get_Hamiltonian(self, k):
        return np.diag(self.get_energies(k))
    
    def initiate_effective(self, nk=101):
        # eps_k is for the numerical stability
        dk = 2*np.pi/nk
        _coeff = np.zeros((self.n_band, self.n_band), dtype=complex)

        for ip in range(nk):
            _sub_hk_start = self.get_subspace_Hamiltonian(dk*ip)
            _sub_hk_end = self.get_subspace_Hamiltonian(dk*(ip+1)) 

            for i in range(self.n_band):
                for j in range(i+1, self.n_band):
                    _sub_1 = _sub_hk_start[i,j]
                    _sub_1 = _sub_1/np.abs(_sub_1)

                    _sub_2 = _sub_hk_end[i,j]
                    _sub_2 = _sub_2/np.abs(_sub_2)

                    _coeff[i,j] = _coeff[i,j] + np.angle(_sub_2/_sub_1)
                    _coeff[j,i] = _coeff[i,j]

            #print(_coeff)
        self.Ns = _coeff/(2*np.pi)
        self.Ns = np.round(self.Ns,2)
        self.Ns = np.real(self.Ns)
        
        return self.Ns #, _coeff

    def get_energies_origin(self, k, nk=51):
        ks = np.linspace(0,k,nk)
        #bands = np.zeros(self.n_band, dtype=complex)

        _band0 = LA.eigvals(self.Hamiltonian_func(0))
        for ik, _k in enumerate(ks):
            _H = self.Hamiltonian_func(_k)
            _band = LA.eigvals(_H)
            indices = np.ones(self.n_band, dtype=int)
            for i, _E in enumerate(_band):
                #print(np.abs(_band0 - _E))
                indices[i] = int(np.argmin(np.abs(_band0 - _E)))

            #print(indices)
            _band0[indices] = _band

        return _band0
    
    def get_energies(self, k, nk=51):
        ks = np.linspace(0, k, nk)
        #bands = np.zeros(self.n_band, dtype=complex)

        _band0 = self.flatten_bands(k=0)
        indices = np.argsort(np.real(_band0))
        _band0 = _band0[indices]
        for ik, _k in enumerate(ks):
            _band = self.flatten_bands(k=_k)
            indices = np.ones(self.n_band, dtype=int)
            for i, _E in enumerate(_band):
                #print(np.abs(_band0 - _E))
                indices[i] = int(np.argmin(np.abs(_band0 - _E)))

            #print(indices)
            _band0[indices] = _band

        return _band0
    
    def flatten_bands(self, k):
        #print(self.ratio)
        k_width = 2*np.pi*(1 - self.compress_ratio)/2
        lower_bound = k_width
        upper_bound = 2*np.pi - k_width

        _E0 = LA.eigvals(self.Hamiltonian_func(0))
        _E0_projection = np.zeros(len(_E0),dtype=complex)
        _E0_start = np.zeros(len(_E0),dtype=complex)

        _n_re, _n_im = np.real(self.axis), np.imag(self.axis)
        n_vector = np.array([_n_re, _n_im])
        n_vector_norm = np.array([-_n_im, _n_re])
        for i, _E in enumerate(_E0):
            _E_re, _E_im = np.real(_E), np.imag(_E)
            _E_vector = np.array([_E_re, _E_im])

            _E0_projection[i] = np.dot(_E_vector, n_vector) + 1j*np.dot((_E_vector - np.dot(_E_vector, n_vector)*n_vector), n_vector_norm)
            _E0_start[i] = np.real(_E0_projection[i])
            
        bands = np.zeros(self.n_band, dtype=complex)
        if k < lower_bound:
            bands = _E0_start + (_E0_projection - _E0_start)*k/k_width

        elif k > upper_bound:
            bands = _E0_projection + (_E0_start - _E0_projection)*(k-upper_bound)/k_width

        else:
            _k = (k-lower_bound)/(upper_bound-lower_bound)*2*np.pi
            _Es = LA.eigvals(self.Hamiltonian_func(_k))
            for i, _E in enumerate(_Es):
                _E_re, _E_im = np.real(_E), np.imag(_E)
                _E_vector = np.array([_E_re, _E_im])

                bands[i] = np.dot(_E_vector, n_vector) + 1j*np.dot((_E_vector - np.dot(_E_vector, n_vector)*n_vector), n_vector_norm)

        return bands

def obtain_non_Hermitian_Hamiltonian(n_band, func, compress_ratio=0.75, axis=1.):
    return NonHamiltonianExtract(func, n_band, compress_ratio=compress_ratio, axis=axis)


def obtain_non_Hermitian_model(n_band, func, compress_ratio=0.75, axis=1.):
    return NonHermitianModel(NonHamiltonianExtract(func, n_band, compress_ratio=compress_ratio, axis=axis))