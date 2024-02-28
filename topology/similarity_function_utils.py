import numpy as np
from .topology_utils import topology_comparator

def similarity_function(model1, model2, perturbations=None, fast_mode=False, epsilon=1e-3):
    if fast_mode:
        winding_matrix_1 = model1.get_winding_matrix()
        winding_matrix_2 = model2.get_winding_matrix()

        if np.all(np.abs(winding_matrix_1-winding_matrix_2) < epsilon):
            return 1
        else:
            return 0

    subspace_models_1 = model1.get_subspace_models()
    subspace_models_2 = model2.get_subspace_models()

    n_subspace_models = len(subspace_models_1)

    if perturbations is None:
        for i in range(n_subspace_models):
            val = topology_comparator(
                subspace_models_1[i], 
                subspace_models_2[i])
            if val == 0:
                return 0
                 
        return 1
    
    else:
        total_val = 1
        for i in range(n_subspace_models):
            val = 0
            for ip, perturbation in enumerate(perturbations):
                _val = topology_comparator(
                    subspace_models_1[i], 
                    subspace_models_2[i], perturbation, n_guess=10)
                val = val or _val

            total_val = total_val and val

        return total_val