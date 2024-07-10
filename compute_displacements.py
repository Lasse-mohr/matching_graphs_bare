import numpy as np
import matplotlib.pyplot as plt
import os

from graph_matching.data_loader.graph_loader import NetworkLoader
from graph_matching.data_loader.graph_loader import compute_mean_network
from graph_matching.fugw_utils.fugw_util import init_and_train_vanilla_fugw
from graph_matching.utils.utils import double_stochastic_array_to_permuation
from graph_matching.utils.utils import compute_displacements


def plot_displacement_hist(displacements, cumulative=True, density=True, nbins=30):

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.get_cmap('tab20b').colors

    xmax = 0
    for i, (model_name, distances) in enumerate(displacements.items()):
        ax.hist(
            distances,
            label=model_name,
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

    return fig, ax


def main(atlas, structure_type, nsubjects=100, plotpath='./figures/displacement'):
    """ 
    ARGUMENTS:
        - nsubjects (int): restricts experiment to given number
            of subjects. Default value is 100 (all subjects)

    """
    loader = NetworkLoader(atlas, structure_type)
    mean_network = compute_mean_network(loader)

    atlas_coord = loader.atlas_coordinates


    #### Perform experiments with vanilla fugw
    results = {}
    for count, (user_id, subject_network) in enumerate(loader):
        if count < nsubjects:
            # calculate the double stochastic matrix
            P = init_and_train_vanilla_fugw(
                                atlas_coord,
                                network_1=subject_network,
                                network_2=mean_network,
                                alpha = 0.5,
                                rho = 1,
                                eps = 1e-5,
                                solver='mm',
                                solver_params={
                                    "nits_bcd": 5,
                                    "tol_bcd": 1e-10,
                                    "tol_uot": 1e-10,
                                },
                            )
            # convert to permutation
            results[count] = double_stochastic_array_to_permuation(
                                                double_stoch_array=P
                                            )

    displacements = compute_displacements(
                                        atlas_coord=atlas_coord,
                                        node_permutations=list(results.values())
                                    )

    displacements_dict = {'fugw':displacements}

    fig, ax = plot_displacement_hist(displacements=displacements_dict, cumulative=True) 
    plt.savefig(os.path.join(plotpath, 'displacement_distances.png'))


if __name__ == '__main__':
    atlas = 'Glasser' #  'Schaefer1000'
    structure_type = 'ses-01'
    plotpath = './figures/displacement'
    nsubjects=1

    main(atlas=atlas, structure_type=structure_type, nsubjects=nsubjects, plotpath=plotpath)

