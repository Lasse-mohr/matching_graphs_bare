# %%
import matplotlib.pyplot as pl
import numpy as np
import ot
import os
from ot.gromov import gromov_wasserstein, fused_gromov_wasserstein, fused_gromov_wasserstein2
import pandas as pd

# %%
import pandas as pd
import numpy as np

def collect_results_to_dataframe(data_dict, results_dict, averaged_array, reference, compare_fgw, analyze_permutations, atlas_switch, type_switch, alpha):
    """
    Collects results into a DataFrame and saves it to a CSV file.

    Parameters:
    - data_dict: Dictionary containing input data items.
    - results_dict: Dictionary containing computed results (Gwg matrices).
    - averaged_array: Averaged array used in computations.
    - reference: Reference matrix for alignment.
    - compare_fgw: Function to compare FGW distances.
    - analyze_permutations: Function to analyze permutations.
    - atlas_switch: The atlas switch value used in the filename.
    - type_switch: The type switch value used in the filename.
    - alpha: The alpha value used in the filename.
    """
    
    # Collect results into a DataFrame
    collated_results = []
    
    for key, item in data_dict.items():
        P_aligned = results_dict[key]
        
        # Compute the quality
        reference_modified = np.abs(reference - 1)
        np.fill_diagonal(reference_modified, 0)
        quality = compute_alignment_quality(averaged_array, item, P_aligned)
        
        # Analyze permutations
        permutation_types = analyze_permutations(P_aligned, reference)
        
        # Collect the data in a dictionary
        row_data = {
            'subject': key,
            'frobenius_quality': quality,
            'self': permutation_types['self'],
            'neighbors': permutation_types['neighbors'],
            'others': permutation_types['others']
        }
        
        collated_results.append(row_data)
    
    # Create the DataFrame in one go
    collated_results_df = pd.DataFrame(collated_results)
    collated_results_df.to_csv(f'/Users/tiyu/LoGML/results/fgw_results_{atlas_switch}_{type_switch}_{alpha}.csv', index=False)

def compute_alignment_quality(G1, G2, P):
    n = G1.shape[0]
    F_opt = np.trace(G1.dot(P).dot(G2.T).dot(P.T))
    F_id = np.trace(G1.dot(np.eye(n)).dot(G2.T).dot(np.eye(n)))
    return F_opt / F_id

def compare_fgw(T, M, C1, C2, alpha):
    """
    Compares the FGW distance of the provided transport plan T with the identity matrix.

    Parameters:
    - T (numpy.ndarray): Transport matrix (permutation matrix).
    - M (numpy.ndarray): Cost matrix.
    - C1 (numpy.ndarray): Cost matrix of the first distribution.
    - C2 (numpy.ndarray): Cost matrix of the second distribution.
    - alpha (float): Weight between the metrics.

    Returns:
    - float: The ratio of FGW distance with T over FGW distance with the identity matrix.
    """
    # Compute FGW distance for the provided transport plan T
    p = ot.uniform(T.shape[0])
    q = ot.uniform(T.shape[1])
    fgw_T, log = fused_gromov_wasserstein2(M, C1, C2, alpha=alpha, log=True, p=p, q=q, verbose=True)
    print(fgw_T)
    print(log)
    

def analyze_permutations(P_aligned, spatial_adj):
    n = P_aligned.shape[0]
    permutation_types = {
        'self': np.sum(np.diag(P_aligned)),
        'neighbors': np.sum(P_aligned * spatial_adj) - np.sum(np.diag(P_aligned)),
        'others': np.sum(P_aligned) - np.sum(P_aligned * spatial_adj)
    }

    return {k: v / n for k, v in permutation_types.items()}


# %%
def load_data(atlas_switch, type_switch):
    data_path = f'/Users/tiyu/LoGML/matching_graphs_spatial_constraints/data/{atlas_switch}'
    all_dir = os.listdir(data_path)
    reference = np.load(f'/Users/tiyu/LoGML/brain_connectomes/derivatives/neigh_matrix_{atlas_switch}.npy')
    reference = np.abs(reference - 1)
    all_dir = [dir for dir in all_dir if 'sub' in dir]
    
    data_dict = {}
    for dir in all_dir:
        if type_switch == 'ses-01':
            structure = np.load(f'/Users/tiyu/LoGML/matching_graphs_spatial_constraints/data/{atlas_switch}/{dir}/{type_switch}/{dir}_{type_switch}_atlas-{atlas_switch}_SC.npy')
            data_dict[dir] = structure
    
    arrays = list(data_dict.values())
    stacked_arrays = np.stack(arrays)
    averaged_array = np.average(stacked_arrays, axis=0)
    
    return data_dict, averaged_array, reference

# %%
atlas_switch ='Glasser' ## 'Glasser' 'Schaefer1000'
type_switch = 'ses-01'
for alpha in [0.01, 0.1, 0.5, 1.0]:
    data_dict, averaged_array, reference = load_data(atlas_switch, type_switch)
    results_dict = np.load(f'/Users/tiyu/LoGML/results/fgw_results_{atlas_switch}_{type_switch}_{alpha}.npy', allow_pickle=True).item()
    collect_results_to_dataframe(data_dict, results_dict, averaged_array, reference, compare_fgw, analyze_permutations, atlas_switch, type_switch, alpha)