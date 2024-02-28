from .non_Hermitian_Hamiltonian import NonHermitianHamiltonianBraid, BraidOperator, NonHermitianHamiltonianTorusKnot
from .generator import GeneratorBraid, obtain_model_braid, obtain_model_torus_knot
from .perturbation_model import sigma_0, sigma_x, sigma_y, sigma_z, PerturbationModel
from .non_Hermitian_model_finite import NonHamiltonianModelFinite, HamiltonianEffective
from .unknot_generator import UnknotGenerator
from .extract_Hamiltonian_utils import obtain_non_Hermitian_Hamiltonian, obtain_non_Hermitian_model

from IPython.display import Markdown, display

def show_braid_words(braid_operators):
    n_operator = len(braid_operators)
    info = ''.join([str(braid_operators[i]) + ' ' for i in reversed(range(n_operator))])
    display(Markdown(info))