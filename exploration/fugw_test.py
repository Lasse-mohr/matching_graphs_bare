import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from fugw.mappings import FUGW

base_dir = os.path.abspath('..')
sys.path.append(base_dir)

from graph_matching.data_loader.graph_loader import NetworkLoader
from graph_matching.data_loader.graph_loader import compute_mean_network

atlas = 'Schaefer1000'
structure_type='ses-01'
loader = NetworkLoader(atlas, structure_type, base_dir=base_dir+'/data_new_struct')

_, network, atlas_coord = next(loader)
atlas_coord = atlas_coord.transpose()
network = network.astype(float)

mean_network, mean_network_atlas_coord = compute_mean_network(loader)

### Normalize
atlas_coord_normalized = atlas_coord / np.linalg.norm(
    atlas_coord, axis=1
).reshape(-1, 1)
network_normalized = network / np.max(network)
mean_network_normalized = mean_network / np.max(mean_network)

alpha = 0.5
rho = 1
eps = 1e-3
mapping = FUGW(alpha=alpha, rho=rho, eps=eps)

_ = mapping.fit(
    atlas_coord_normalized,
    atlas_coord_normalized,
    source_geometry=network_normalized,
    target_geometry=mean_network_normalized,
    solver="sinkhorn",
    solver_params={
        "nits_bcd": 5,
        "tol_bcd": 1e-10,
        "tol_uot": 1e-10,
    },
    verbose=False,
)

pi = mapping.pi.numpy()
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_title("Transport plan", fontsize=20)
ax.set_xlabel("target vertices", fontsize=15)
ax.set_ylabel("source vertices", fontsize=15)
im = plt.imshow(np.log(pi), cmap="magma")
#im = plt.imshow(pi, cmap="magma")
plt.colorbar(im, ax=ax, shrink=0.8)
plt.show()

