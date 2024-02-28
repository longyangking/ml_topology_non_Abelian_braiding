import numpy as np
import numpy.linalg as LA
from .non_Hermitian_Hamiltonian import NonHermitianHamiltonianBraid
import copy
import itertools
import os

class HamiltonianEffective:
    def __init__(self, n_band, braid_operators):
        self.n_band = n_band
        self.braid_operators = braid_operators
        #self.n_operator = len(self.braid_operators)
        self.hamiltonian = NonHermitianHamiltonianBraid(n_band=n_band,
                                                        braid_operators=braid_operators) 

    # def initiate(self, nk=101):
    #     '''
    #     Calculate the coefficient of Fourier transformation
    #     '''
    #     bands = np.zeros((nk, self.n_band),dtype=complex)
    #     ks = np.linspace(0, 2*np.pi,nk) # all frequency will be double !!!

    #     for ik, k in enumerate(ks):
    #         bands[ik] = self.hamiltonian.get_energies(k)
        
    #     #_Fourier_bands_raw = np.zeros((self.n_band, nk), dtype=complex)

    def get_Hamiltonian(self, k):
        Es = self.hamiltonian.get_energies(k)
        coefficients = dict()

        for icoeff in range(self.n_band+1):
            combinations = list(itertools.combinations(range(self.n_band), icoeff))
            
            _val = 0
            for comb in combinations:
                _v = 1
                for index in comb:
                    _v = _v*Es[index]
                _val = _val + _v
                
            coefficients[icoeff] = _val

        Hamiltonian = np.zeros((self.n_band, self.n_band), dtype=complex)
        for i in range(self.n_band-1):
            Hamiltonian[i+1, i] = 1.
            
        for i in range(self.n_band):
            Hamiltonian[0,i] = coefficients[i+1]*((-1)**(i))

        return Hamiltonian

    def get_n_band(self):
        return self.n_band
    
    def get_Fourier_coefficients(self, nk=101, order=None):
        Fourier_coefficient_list = list()
        term_values = np.zeros((self.n_band, nk),dtype=complex)
        ks = np.linspace(0,2*np.pi,nk)

        for ik, k in enumerate(ks):
            for icoeff in range(self.n_band):
                Hamiltonian = self.get_Hamiltonian(k)
                term_values[icoeff, ik] = Hamiltonian[0, icoeff]

        for icoeff in range(self.n_band):
            _Fourier_coefficients = dict()
            _Fourier_coeff = np.fft.fft(term_values[icoeff, :])/nk
            
            for index in range(nk):
                if index < nk/2:
                    _Fourier_coefficients[index] = _Fourier_coeff[index]
                else:
                    _Fourier_coefficients[index-nk] = _Fourier_coeff[index]

            Fourier_coefficient_list.append(_Fourier_coefficients) 

        return Fourier_coefficient_list
    
    def get_Hamiltonian_Fourier(self, k, epsilon=None, order=None, constant_term=1.):
        Fourier_coefficients = self.get_Fourier_coefficients()
        Hamiltonian = np.zeros((self.n_band, self.n_band), dtype=complex)

        for i in range(self.n_band-1):
            Hamiltonian[i+1, i] = constant_term

        for icoeff in range(self.n_band):
            _Fourier_coeffs = Fourier_coefficients[icoeff]
            for key in _Fourier_coeffs:
                _val = _Fourier_coeffs[key]*np.exp(1j*key*k)/(constant_term**icoeff)

                flag = True
                if order is not None:
                    if np.abs(key) > order:
                        flag = False     
                elif epsilon is not None:
                    if np.abs(_val) < epsilon:
                        flag = False

                if flag:
                    Hamiltonian[0, icoeff] = Hamiltonian[0, icoeff] + _val

        return Hamiltonian
    
    def get_energies_Fourier(self, k, epsilon=None, order=None, constant_term=1.):
        Hamiltonian = self.get_Hamiltonian_Fourier(k, epsilon=epsilon, order=order, constant_term=1.)
        values, _ = LA.eig(Hamiltonian)
        return values

    def __call__(self, k):
        return self.get_Hamiltonian(k)
    
    def get_winding_matrix(self):
        return self.hamiltonian.initiate_effective()
    
    def get_energies(self, k):
        #k = k%b # contraint it in the first BZ
        values, _ = LA.eig(self.get_Hamiltonian(k))
        return values
          
    def get_winding_number(self, E_ref, nk=101):
        phase = 0
        ks = np.linspace(0, 2*np.pi, nk)

        _energy_prev = None
        for k in ks:
            _energy = LA.det(self.get_Hamiltonian(k) - E_ref*np.identity(self.n_band))
            #print(_energy)
            if _energy_prev is None:
                _energy_prev = _energy/np.abs(_energy)
            else:
                _energy = _energy/np.abs(_energy)
                phase = phase + np.angle(_energy/_energy_prev)
                _energy_prev = _energy
        
        winding_number = np.round(phase/(2*np.pi))
        return winding_number
    
    def get_n_degeneracy(self):
        E_refs = 1.5 + np.arange(self.n_band-1)
        num = np.sum([np.abs(self.get_winding_number(E_ref)) for E_ref in E_refs])
        return num
    
    def save_band(self, filename, n_k=101):
        ks = np.linspace(0,2*np.pi, n_k)
        Es = np.zeros((n_k, 2*self.n_band))
        for i, k in enumerate(ks):
            _E = self.hamiltonian.get_energies(k)
            for n in range(self.n_band):
                Es[i, 2*n] = np.real(_E[n])
                Es[i, 2*n+1] = np.imag(_E[n])

        np.savetxt(filename, Es)
    
    def save_band_coefficients(self, directory_name, filename):
        _exist_folder = os.path.exists('./{directory_name}'.format(directory_name=directory_name))
        if not _exist_folder:
            os.mkdir('./{directory_name}'.format(directory_name=directory_name))

        _exit_filename_folder = os.path.exists('./{directory_name}/{filename}'.format(
            directory_name=directory_name, filename=filename))
        if not _exit_filename_folder:
            os.mkdir('./{directory_name}/{filename}'.format(
                directory_name=directory_name, filename=filename))

        Fourier_coefficient_list = self.get_Fourier_coefficients()
        n_term = len(Fourier_coefficient_list)

        for iterm in range(n_term):
            Fourier_coefficient = Fourier_coefficient_list[iterm]
            coeffs = np.zeros((len(Fourier_coefficient),3))
            for i, key in enumerate(Fourier_coefficient):
                coeffs[i,0] = int(key)
                coeffs[i,1] = np.real(Fourier_coefficient[key])
                coeffs[i,2] = np.imag(Fourier_coefficient[key])
                
            np.savetxt('./{directory_name}/{filename}/{iterm}.txt'.format(
                directory_name=directory_name, filename=filename, iterm=iterm), coeffs)

    #bands = [np.sum([vF*np.exp(1j*iF*k) for iF, vF in enumerate(self.Fourier_bands[ib])]) for ib in range(self.n_band)]    
    # if order is not None:
    #         bands = [np.sum([self.Fourier_bands[ib][key]*np.exp(1j*key/2*k) for key in self.Fourier_bands[ib] if np.abs(key/2) <= order ]) for ib in range(self.n_band)]
    #     else:
    #         bands = [np.sum([self.Fourier_bands[ib][key]*np.exp(1j*key/2*k) for key in self.Fourier_bands[ib]]) for ib in range(self.n_band)]
        

class NonHamiltonianModelFinite:
    def __init__(self, n_unit_cell, Hamiltonian_effective, order=None, epsilon=None, constant_term=1.):
        self.n_unit_cell = n_unit_cell
        self.Hamiltonian_effective = Hamiltonian_effective
        self.Hamiltonian_finite = None
        self.order = order
        self.epsilon = epsilon
        self.constant_term = constant_term
        self.initiate(self.order, self.epsilon, self.constant_term)

    def get_n_site(self):
        n_band = self.Hamiltonian_effective.get_n_band()
        return self.n_unit_cell*n_band
        
    def initiate(self, order=None, epsilon=None, constant_term=1., Hermitian_perturbation=0.):
        n_band = self.Hamiltonian_effective.get_n_band()
        Hamiltonian_finite_list = dict()
        Fourier_coefficients = self.Hamiltonian_effective.get_Fourier_coefficients()

        for i_cell in range(self.n_unit_cell):
            index = i_cell*n_band
            for i in range(n_band-1):
                Hamiltonian_finite_list[(index+i+1, index+i)] = constant_term

            for icoeff in range(n_band):
                _Fourier_coeffs = Fourier_coefficients[icoeff]
                for key in _Fourier_coeffs:
                    _val = _Fourier_coeffs[key]

                    flag = True
                    if order is not None:
                        if np.abs(key) > order:
                            flag = False     
                    elif epsilon is not None:
                        if np.abs(_val) < epsilon:
                            flag = False

                    if flag:
                        Hamiltonian_finite_list[(index, index + icoeff + key*n_band)] = _val/(constant_term**icoeff)

        n_site = self.n_unit_cell*n_band
        Hamiltonian_finite = np.zeros((n_site, n_site), dtype=complex)
        for position in Hamiltonian_finite_list:
            i,j = position
            if (i >=0) and (i<n_site) and (j >=0) and (j<n_site):
                Hamiltonian_finite[i,j] = Hamiltonian_finite_list[position]

        Hamiltonian_Hermitian = np.zeros((n_site, n_site))
        for i in range(n_site):
            if (i-1) >= 0:
                Hamiltonian_Hermitian[i,i-1] = Hermitian_perturbation
            if (i+1) < n_site:
                Hamiltonian_Hermitian[i,i+1] = Hermitian_perturbation

        Hamiltonian_finite = Hamiltonian_finite + Hamiltonian_Hermitian

        self.Hamiltonian_finite = Hamiltonian_finite

    def get_Hamiltonian(self):
        return self.Hamiltonian_finite

    def get_eigensystems(self, decimals=None, is_sort=True):
        values, vectors = LA.eig(self.Hamiltonian_finite)
        vectors = np.transpose(vectors)

        if decimals is not None:
            values = np.round(values, decimals=decimals)

        if is_sort:
            indices = np.argsort(-np.real(values))
            values = values[indices]
            vectors = vectors[indices]

        return values, vectors
    
    def get_degenercy(self, epsilon=1e-3, n_center=None, n_range=20):
        values, _ = self.get_eigensystems()
        n_site = self.get_n_site()
        
        #n_center = int(n_site/2)
        if n_center is None:
            n_center = int(np.random.choice(range(n_range, n_site-n_range), 1))

        degenercy = 0
        value = values[n_center]
        for i in range(n_center-n_range, n_center+n_range):
            _val = values[i]
            if np.abs(value - _val) < epsilon:
                degenercy = degenercy + 1

        return degenercy
    
    def get_number_branches(self, epsilon=1e-3):
        values, vectors = self.get_eigensystems()
        n_site = self.get_n_site()
        
        branches = list()
        branches.append(0)
        
        for i in range(1, n_site):
            psi = vectors[i]
            psi = psi/LA.norm(psi)
            
            flag = True
            for ib in branches:
                psib = vectors[ib]
                psib = psib/LA.norm(psib)
                if np.abs(1 - np.abs(np.dot(np.conjugate(psib), psi))) < epsilon:
                    flag = False

            if flag:
                branches.append(i)

        return len(branches), branches
                
    