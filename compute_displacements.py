import numpy as np
import matplotlib.pyplot as plt
import os

from graph_matching.data_loader.graph_loader import NetworkLoader
from graph_matching.data_loader.graph_loader import compute_mean_network
from graph_matching.fugw_utils.fugw_util import init_and_train_fugw
from graph_matching.utils.utils import double_stochastic_array_to_permuation
from graph_matching.utils.utils import compute_displacements


def plot_displacement_hist(displacements, cumulative=True, density=True, nbins=30, logy=False):

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.get_cmap('tab20b').colors

    xmax = 0
    for i, (experiment_name, distances) in enumerate(displacements.items()):
        ax.hist(
            distances,
            label=experiment_name,
            bins=nbins,
            histtype='step',
            color=colors[i],
            cumulative=cumulative,
            density=density,
            linewidth=3,
            )
        xmax = np.max([np.max(distances), xmax])

    ax.set_xlim([-1, xmax])
    ax.set_xlabel('Distance of Permutation', fontsize=20)
    ax.set_ylabel('Cumulative Frequency', fontsize=20)

    plt.legend(fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)

    if logy:
        ax.set_yscale('log')

    return fig, ax


def compute_permutations(atlas, structure_type, model='fugw', kwargs={}):
    loader = NetworkLoader(atlas, structure_type)
    mean_network = compute_mean_network(loader)

    atlas_coord = loader.atlas_coordinates

    if model == 'fugw':
        init_and_train_fun = init_and_train_fugw
    elif model == 'rigid_alignment':
        # init_and_train_fun = init_and_train_rigid_alignment
        pass

    #### Perform experiments with vanilla fugw
    permutation_dict = {}
    for count, (user_id, subject_network) in enumerate(loader):
        if count < nsubjects:
            # calculate the double stochastic matrix
            P = init_and_train_fun(
                                atlas_coord,
                                network_1=subject_network,
                                network_2=mean_network,
                                **kwargs
                            )
            # convert to permutation
            permutation_dict[user_id] = double_stochastic_array_to_permuation(
                                                double_stoch_array=P
                                            )
    return permutation_dict


def main(atlas, structure_type, nsubjects=100, plotpath='./figures/displacement', experiments={}, log_y_scale=False):
    """ 
    ARGUMENTS:
        - nsubjects (int): restricts experiment to given number
            of subjects. Default value is 100 (all subjects)

    """
    atlas_coord = NetworkLoader(atlas, structure_type).atlas_coordinates

    displacements_dict = {}

    for experiment_name, experiment in experiments.items():
        model = experiment['model']
        kwargs = experiment['kwargs']
        permutation_dict = compute_permutations(
                                                atlas=atlas,
                                                structure_type=structure_type,
                                                model=model,
                                                kwargs=kwargs
                                                )
        displacements = compute_displacements(
            atlas_coord=atlas_coord, 
            node_permutations=list(permutation_dict.values())
            )
        displacements_dict[experiment_name] = displacements

    fig, ax = plot_displacement_hist(
        displacements=displacements_dict,
        cumulative=False,
        logy=log_y_scale,
        ) 
    plt.savefig(os.path.join(plotpath, 'displacement_distances.png'))


if __name__ == '__main__':
    atlas = 'Glasser' #  'Schaefer1000'
    structure_type = 'ses-01'
    plotpath = './figures/displacement'
    nsubjects=3

    experiments = {
        #### EXPERIMENT 1 #####################################
        'fugw: no geometry': {
                        'model':'fugw', 
                        'kwargs' : {
                                'alpha': 0,
                                'rho': 1,
                                'eps': 1e-5,
                                'solver': 'mm',
                                'solver_params':{
                                                "nits_bcd": 5,
                                                "tol_bcd": 1e-10,
                                                "tol_uot": 1e-10,
                                                }
                                },
                            },
        #### EXPERIMENT 2 #####################################
        'fugw: no network': {
                        'model':'fugw', 
                        'kwargs' : {
                                'alpha': 1,
                                'rho': 1,
                                'eps': 1e-5,
                                'solver': 'mm',
                                'solver_params':{
                                                "nits_bcd": 5,
                                                "tol_bcd": 1e-10,
                                                "tol_uot": 1e-10,
                                                }
                                },
                            },
        #### EXPERIMENT 3 #####################################
        'fugw': {
                'model':'fugw', 
                'kwargs' : {
                        'alpha': 0.5,
                        'rho': 1,
                        'eps': 1e-5,
                        'solver': 'mm',
                        'solver_params':{
                                        "nits_bcd": 5,
                                        "tol_bcd": 1e-10,
                                        "tol_uot": 1e-10,
                                        }
                        },
                }
        }

    main(atlas=atlas,
         structure_type=structure_type,
         nsubjects=nsubjects,
         plotpath=plotpath,
         experiments=experiments,
         log_y_scale=True
        )

