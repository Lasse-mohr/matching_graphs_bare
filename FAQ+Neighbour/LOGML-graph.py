import numpy as np
from scipy.optimize import linear_sum_assignment
import nibabel as nib
import nilearn
print(nilearn.__version__)
from nilearn import datasets
import pandas as pd

def reordervec(vec, neworder):
    newvec = []
    for i in range(len(neworder)):
        newvec.append(vec[np.where(neworder == i)[0][0]])
    return np.array(newvec)
def reordermat(mat, neworder):
    temp = reordervec(mat, neworder)
    return reordervec(temp.T, neworder).T
def path_dir(path, dtype="npy"):
    fname = os.listdir(path)
    fname = [x for x in fname if x.endswith(dtype)]
    fname.sort()
    out_path = [os.path.join(path, name) for name in fname]
    return out_path

def dict_to_matrix(data):
    labels = sorted(data.keys())
    return np.array([data[label] for label in labels])


def df_to_matrix(df):
    label_col = df.columns[0]
    unique_labels = df[label_col].unique()
    grouped = [df[df[label_col] == label].drop(label_col, axis=1).values.flatten().tolist()
               for label in unique_labels]
    return np.array(grouped)

def dict_to_dataframe(data):
    return pd.DataFrame.from_dict(data, orient='index').reset_index()
def extract_glasser_360_coordinates(nifti_file, nodeindex_file, nodenames_file):
    # Load the NIfTI file
    img = nib.load(nifti_file)
    data = img.get_fdata()
    affine = img.affine

    # Load the Nodeindex file
    node_indices = np.loadtxt(nodeindex_file, dtype=int)

    # Load the nodenames file
    with open(nodenames_file, 'r') as f:
        node_names = [line.strip() for line in f]

    # Create a dictionary to store coordinates for each region
    region_coordinates = {}

    for index, name in zip(node_indices, node_names):
        # Find voxel coordinates where the index appears
        voxel_coords = np.array(np.where(data == index)).T

        if len(voxel_coords) > 0:
            # Convert voxel coordinates to world coordinates
            world_coords = nib.affines.apply_affine(affine, voxel_coords)

            # Calculate the mean coordinate (center of mass)
            mean_coord = np.mean(world_coords, axis=0)

            # Store in dictionary
            region_coordinates[name] = mean_coord
        else:
            print(f"Warning: No voxels found for region {name} (index {index})")

    # Convert to DataFrame for easy viewing and further processing
    df = pd.DataFrame.from_dict(region_coordinates, orient='index', columns=['x', 'y', 'z'])
    df.index.name = 'Region'
    df.reset_index(inplace=True)

    return df
#%%
nifti_file = '/Users/sonmjack/Downloads/LOGML/glasser360MNI.nii.gz'
nodeindex_file = '/Users/sonmjack/Downloads/LOGML/glasser360NodeIndex.1D'
nodenames_file = '/Users/sonmjack/Downloads/LOGML/glasser360NodeNames.txt'

coordinates_df = extract_glasser_360_coordinates(nifti_file, nodeindex_file, nodenames_file)
coordinates_Gla = df_to_matrix(coordinates_df)
SC_temple_Gla = np.load('/Users/sonmjack/Downloads/LOGML/brain_connectomes/derivatives/neigh_matrix_Glasser.npy')
#%%
SC_temple_Schaefer = np.load('/Users/sonmjack/Downloads/LOGML/brain_connectomes/derivatives/neigh_matrix_Schaefer1000.npy')
#%%
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7, resolution_mm=1, data_dir=None, base_url=None, resume=True, verbose=1)
# Load the atlas image
atlas_img = nib.load(atlas['maps'])

# Get the atlas data and affine transformation
atlas_data = atlas_img.get_fdata()
affine = atlas_img.affine
# Get unique region labels (excluding 0, which is typically background)
region_labels = np.unique(atlas_data)[1:]

# Initialize a dictionary to store coordinates for each region
coordinates_Sch = {}

for label in region_labels:
    # Find voxel coordinates where the label appears
    voxel_coords = np.array(np.where(atlas_data == label)).T

    # Convert voxel coordinates to world coordinates
    world_coords = nib.affines.apply_affine(affine, voxel_coords)

    # Calculate the mean coordinate (center of mass)
    mean_coord = np.mean(world_coords, axis=0)

    # Store in dictionary
    coordinates_Sch[label] = mean_coord

#coordinates_Schaefer_data = dict_to_dataframe(coordinates_Sch).set_index('index')
coordinates_Schaefer = dict_to_matrix(coordinates_Sch)
#%%
SC1 = np.load("/Users/sonmjack/Downloads/LOGML/brain_connectomes/derivatives/sub-100206/connectome/sub-100206_ses-01_atlas-Schaefer1000_SC.npy")
SC2 = np.load("/Users/sonmjack/Downloads/LOGML/brain_connectomes/derivatives/sub-101107/connectome/sub-101107_ses-01_atlas-Schaefer1000_SC.npy")
#%%
FC1 = np.load("/Users/sonmjack/Downloads/LOGML/brain_connectomes/derivatives/sub-101107/connectome/sub-101107_task-rest_run-1_atlas-Schaefer1000_desc-lrrl_FC.npy")
#%%
def sinkhorn_knopp(P, r,c,n,max_iter=50, tolerance=1e-6):
    P_n = P.copy()
    P_n /= P_n.sum()
    u = np.zeros(n)
    while np.max(np.abs(u - P_n.sum(1))) > tolerance:
        u = P_n.sum(1)
        P_n *= (n / P_n.sum(1)).reshape((-1, 1))
        P_n *= (n / P_n.sum(0)).reshape((-1, 1))
    return P_n

def faq_matching(G1, G2, P0, num_iters=30, tolerance =1e-6,lam =1   ):
    P = P0.copy()
    n, m = G1.shape[0], G2.shape[0]
    r = np.ones(n) / n
    c = np.ones(m) / m
    for _ in range(num_iters):
        grad = -(G1.dot(P).dot(G2.T))-(G1.T.dot(P).dot(G2))
        M = np.exp(lam*grad)
        P_new = sinkhorn_knopp(M,r,c,n=n)
        if np.max(np.abs(P_new - P)) < tolerance:
            break
        P = P_new

    row_ind, col_ind = linear_sum_assignment(-P)
    P_final = np.zeros_like(P)
    P_final[row_ind, col_ind] = 1

    return P_final


def graph_alignment(G1, G2, initialization='spatial'):
    n = G1.shape[0]

    if initialization == 'spatial':
        #spatial_adj = SC_temple_Schaefer
        P0 = SC_temple_Schaefer
        #P0 = spatial_adj / spatial_adj.sum(axis=1)[:, np.newaxis]
    elif initialization == 'identity':
        P0 = np.eye(n)
    elif initialization == 'barycenter':
        P0 = np.ones((n, n))
    else:  # random
        P0 = np.random.rand(n, n)

    P_aligned = faq_matching(G1, G2, P0)

    return P_aligned


def compute_alignment_quality(G1, G2, P):
    F_opt = np.trace(G1.dot(P).dot(G2.T).dot(P.T))
    F_id = np.trace(G1.dot(np.eye(n)).dot(G2.T).dot(np.eye(n)))
    return F_opt / F_id


def analyze_permutations(P_aligned, spatial_adj):
    n = P_aligned.shape[0]
    permutation_types = {
        'self': np.sum(np.diag(P_aligned)),
        'neighbors': np.sum(P_aligned * spatial_adj) - np.sum(np.diag(P_aligned)),
        'others': np.sum(P_aligned) - np.sum(P_aligned * spatial_adj)
    }

    return {k: v / n for k, v in permutation_types.items()}

# Example usage
n = SC1.shape[0]  # number of brain regions
G1 = SC1  # Placeholder for patient 1's connectivity matrix
G2 = SC2  # Placeholder for patient 2's connectivity matrix
spatial_adj = SC_temple_Schaefer
P_aligned = graph_alignment(G1, G2, initialization='spatial')
alignment_quality = compute_alignment_quality(G1, G2, P_aligned)

print(f"Alignment quality: {alignment_quality}")

# Analyze permutations
permutation_rates = {
    'self': np.diag(P_aligned),
    'neighbors': np.sum(P_aligned * spatial_adj, axis=1) - np.diag(P_aligned),
    'others': np.sum(P_aligned, axis=1) - np.sum(P_aligned * spatial_adj, axis=1)
}

permutation_types = analyze_permutations(P_aligned, spatial_adj)

print("Permutation types:")
for perm_type, ratio in permutation_types.items():
    print(f"{perm_type}: {ratio:.2%}")


#%%
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update(plt.rcParamsDefault)
import seaborn as sns
import matplotlib.colors as colors

divnorm = colors.TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0)
display1 = plt.imshow(EC_all_infant1, cmap='RdBu_r', norm=divnorm)

# 对每行进行输出
plt.colorbar()
ticks=[3.5,5.5,8.5,10.5,13.5,15.5]
plt.xticks(ticks)
plt.yticks(ticks)
plt.grid( color='0.45',linewidth =1.5 )