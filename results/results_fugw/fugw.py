import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import ot
from tqdm import tqdm
from ot.gromov import fused_gromov_wasserstein
import argparse

def compute_gwg(dir, atlas_switch, type_switch, averaged_array, reference, alpha):
    structure = np.load(f'/Users/tiyu/LoGML/matching_graphs_spatial_constraints/data/{atlas_switch}/{dir}/{type_switch}/{dir}_{type_switch}_atlas-{atlas_switch}_SC.npy')
    n = structure.shape[0]
    n2 = reference.shape[0]
    p = ot.unif(n)
    q = ot.unif(n2)
    C1 = structure
    C2 = averaged_array
    M = reference
    Gwg, logw = fused_gromov_wasserstein(M, C1, C2, p, q, loss_fun='square_loss', alpha=alpha, verbose=False, log=True)
    return (dir, Gwg)

def main(atlas_switch, type_switch, alpha, max_workers):
    data_path = f'/Users/tiyu/LoGML/matching_graphs_spatial_constraints/data/{atlas_switch}'
    all_dir = os.listdir(data_path)
    reference = np.load(f'/Users/tiyu/LoGML/brain_connectomes/derivatives/neigh_matrix_{atlas_switch}.npy')
    reference = np.abs(reference - 1)
    ## Change the diagonals to be 0
    np.fill_diagonal(reference, 0)
    print(reference)
    all_dir = [dir for dir in all_dir if 'sub' in dir]
    
    data_dict = {}
    for dir in all_dir:
        if type_switch == 'ses-01':
            structure = np.load(f'/Users/tiyu/LoGML/matching_graphs_spatial_constraints/data/{atlas_switch}/{dir}/{type_switch}/{dir}_{type_switch}_atlas-{atlas_switch}_SC.npy')
            data_dict[dir] = structure
    
    arrays = list(data_dict.values())
    stacked_arrays = np.stack(arrays)
    averaged_array = np.average(stacked_arrays, axis=0)
    
    results_dict = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_gwg, dir, atlas_switch, type_switch, averaged_array, reference, alpha): dir for dir in all_dir}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            dir, Gwg = future.result()
            max_indices = np.argmax(Gwg, axis=1)
            Gwg_bin = np.zeros_like(Gwg)
            Gwg_bin[np.arange(Gwg.shape[0]), max_indices] = 1
            results_dict[dir] = Gwg_bin
    
    np.save(f'/Users/tiyu/LoGML/results/fgw_results_{atlas_switch}_{type_switch}_{alpha}.npy', results_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run FGW computations in parallel.')
    parser.add_argument('--atlas_switch', type=str, required=True, help='Atlas switch value (e.g., Glasser).')
    parser.add_argument('--type_switch', type=str, required=True, help='Type switch value (e.g., ses-01).')
    parser.add_argument('--alpha', type=float, required=True, help='Alpha value for FGW.')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of workers for parallel processing.')
    
    args = parser.parse_args()
    main(args.atlas_switch, args.type_switch, args.alpha, args.max_workers)
