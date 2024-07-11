from fugw.mappings import FUGW
import numpy as np

def init_and_train_fugw(
                    atlas_coord,
                    network_1,
                    network_2,
                    alpha = 0.5,
                    rho = 1,
                    eps = 1e-5,
                    solver='mm',
                    solver_params={
                        "nits_bcd": 5,
                        "tol_bcd": 1e-10,
                        "tol_uot": 1e-10,
                    },
    ):
    """ 
    Initialize the fugw model. Normalize network and coordinate properly 
    for numerical stability. Fits the model to the network and 
    atlas coordinates and returns the double stochastic matrix 
    """

    mapping = FUGW(alpha=alpha, rho=rho, eps=eps)
    atlas_coord_normalized = atlas_coord / np.linalg.norm(
        atlas_coord, axis=0
    )
    network_1_normalized = network_1 / np.max(network_1)
    network_2_normalized = network_2 / np.max(network_2)

    _ = mapping.fit(
        atlas_coord_normalized.transpose(),
        atlas_coord_normalized.transpose(),
        source_geometry=network_1_normalized,
        target_geometry=network_2_normalized,
        solver=solver,
        solver_params=solver_params,
        verbose=True,
    )

    return mapping




def compute_ratio_fugw(subject_network, mean_network, atlas_coord, kwargs):
    res = init_and_train_fugw(
                            atlas_coord=atlas_coord,
                            network_1=subject_network,
                            network_2=mean_network,
                            **kwargs
                            )

    loss_values = res.loss['total']
    ratio = loss_values[0]/loss_values[-1]
    return ratio
