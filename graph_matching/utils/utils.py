from scipy.optimize import linear_sum_assignment
import numpy as np

def double_stochastic_array_to_permuation(double_stoch_array):
    """ 
    Function for finding the permutation matrix closest to given
    double stochastic matrix.

    Only returns one half of the permutation. The other half is the 
    identity and can be calculated using np.arange(col.shape[0])
    """
    _, col = linear_sum_assignment(double_stoch_array, maximize=True)
    return col

def compute_displacements(atlas_coord, node_permutations: list):
    """
    Computes the distances that each atlas node is permuted by
    a series of permutations.

    INPUT:
        - atlas_coord (np.array)  num subject X 3 array with
            coordinates of each node in the atlas
        - node_permutations (list[np.array()]): list of arrays, each array
            contains the permutation found for a single person

    OUT:
        - numpy array with the distances of how far away each node
            was permutted for each person
    """
    return np.concatenate(
                [
                    np.linalg.norm(atlas_coord - atlas_coord[node_permutation, :], axis=1)
                    for node_permutation in node_permutations 
                ]
            )


