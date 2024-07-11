import numpy as np
from scipy.optimize import linear_sum_assignment
import nibabel as nib
import nilearn
print(nilearn.__version__)
from nilearn import datasets
import pandas as pd
import os


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

SC_temple_Gla = np.load('/Users/sonmjack/Downloads/LOGML/brain_connectomes/derivatives/neigh_matrix_Glasser.npy')
#%%
SC_temple_Schaefer = np.load('/Users/sonmjack/Downloads/LOGML/brain_connectomes/derivatives/neigh_matrix_Schaefer1000.npy')


#%%
def sinkhorn_knopp(P, n, max_iter=50, tolerance=1e-6):
    P_n = P.copy()
    P_n /= P_n.sum()
    u = np.zeros(n)
    while np.max(np.abs(u - P_n.sum(1))) > tolerance:
        u = P_n.sum(1)
        P_n *= (1 / P_n.sum(1)).reshape((-1, 1))
        P_n *= (1 / P_n.sum(0)).reshape((-1, 1))
        # P_n *= (n / P_n.sum(1)).reshape((-1, 1))
        # P_n *= (n / P_n.sum(0)).reshape((-1, 1))
    return P_n

def doubly_stochastic(P, tol=1e-10):
    # Adapted from @btaba implementation
    # https://github.com/btaba/sinkhorn_knopp
    # of Sinkhorn-Knopp algorithm
    # https://projecteuclid.org/euclid.pjm/1102992505

    max_iter = 1000
    c = 1 / P.sum(axis=0)
    r = 1 / (P @ c)
    P_eps = P

    for it in range(max_iter):
        if ((np.abs(P_eps.sum(axis=1) - 1) < tol).all() and
                (np.abs(P_eps.sum(axis=0) - 1) < tol).all()):
            # All column/row sums ~= 1 within threshold
            break

        c = 1 / (r @ P)
        r = 1 / (P @ c)
        P_eps = r[:, None] * P * c

    return P_eps
# def faq_matching(G1, G2, P0, num_iters=30, tolerance =1e-6,lam =1   ):
#     P = P0.copy()
#     n, m = G1.shape[0], G2.shape[0]
#     r = np.ones(n) / n
#     c = np.ones(m) / m
#     for _ in range(num_iters):
#         grad = -(G1.dot(P).dot(G2.T))-(G1.T.dot(P).dot(G2))
#         M = np.exp(lam*grad)
#         P_new = sinkhorn_knopp(M,r,c,n=n)
#         if np.max(np.abs(P_new - P)) < tolerance:
#             break
#         P = P_new
#
#     row_ind, col_ind = linear_sum_assignment(-P)
#     P_final = np.zeros_like(P)
#     P_final[row_ind, col_ind] = 1
#
#     return P_final

class FAQGraphMatcher:
    def __init__(self,P0: np.ndarray, A: np.ndarray, B: np.ndarray, epsilon: float = 1e-4, max_iter: int = 100):
        self.A = A
        self.B = B
        self.n = A.shape[0]
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.P0 = P0

    def doubly_stochastic(self, P: np.ndarray,  tolerance=1e-4):
        """Apply Sinkhorn balancing to make the matrix doubly stochastic."""
        P /= P.sum()
        u = np.zeros(n)
        while np.max(np.abs(u - P.sum(1))) > tolerance:
            u = P.sum(1)
            P /= P.sum(axis=1, keepdims=True)
            P /= P.sum(axis=0, keepdims=True)
        return P

    def compute_gradient(self, P: np.ndarray):
        """Step 1: Compute the gradient of f(P)."""
        return -np.dot(self.A, np.dot(P, self.B.T)) - np.dot(self.A.T, np.dot(P, self.B))

    def compute_search_direction(self, grad: np.ndarray):
        """Step 2: Compute the search direction Q."""
        row_ind, col_ind = linear_sum_assignment(grad)
        Q = np.zeros_like(grad)
        Q[row_ind, col_ind] = 1
        return Q - grad

    def compute_step_size(self, P: np.ndarray, Q: np.ndarray):
        """Step 3: Compute the step size alpha."""

        def f(alpha):
            return self.objective_function(P + alpha * Q)

        # Simple line search for alpha
        alphas = np.linspace(0, 1, 100)
        values = [f(alpha) for alpha in alphas]
        return alphas[np.argmin(values)]

    def objective_function(self, P: np.ndarray):
        """Compute the objective function f(P) = -trace(A*P*B^T*P^T)."""
        return -np.trace(np.dot(np.dot(self.A, np.dot(P, self.B.T)), P.T))

    def update_P(self, P: np.ndarray, alpha: float, Q: np.ndarray):
        """Step 4: Update P."""
        return P + alpha * Q

    def project_to_permutation(self, P: np.ndarray):
        """Project onto the set of permutation matrices."""
        row_ind, col_ind = linear_sum_assignment(-P)
        P_perm = np.zeros_like(P)
        P_perm[row_ind, col_ind] = 1
        self.doubly_stochastic(P)
        return P_perm

    def match(self):
        """Main method to perform graph matching."""
        P = self.P0
        i = 0
        for _ in range(self.max_iter):
            P_prev = P.copy()

            grad = self.compute_gradient(P)
            Q = self.compute_search_direction(grad)
            alpha = self.compute_step_size(P, Q)
            P = self.update_P(P, alpha, Q)
            i = i+1
            print('finish'+str(i))
            if np.linalg.norm(P - P_prev, 'fro') < self.epsilon:
                break

        # Final projection to permutation matrix
        P_final = self.project_to_permutation(P)
        return P_final

#%%
def set_one_in_zero_rows(matrix):
    # 找出全为0的行
    zero_rows = np.where(~matrix.any(axis=1))[0]

    # 对每个全为0的行进行处理
    for row in zero_rows:
        # 随机选择该行的一个位置
        random_col1 = row+1
        # 将选中的位置设置为1
        matrix[row, random_col1] = 1
        matrix[random_col1, row] = 1

        random_col2 = row-1
        # 将选中的位置设置为1
        matrix[row, random_col2] = 1
        matrix[random_col2, row] = 1
    return matrix
#%%
def generate_adjacency_matrix(vector):
    n = vector.shape[0]
    matrix = np.zeros((n, n), dtype=int)

    for i, v in enumerate(vector):
        matrix[i][v] = 1
        matrix[v][i] = 1  # 对称设置

    return matrix
#%%
def graph_alignment(G1, G2, initialization='spatial'):
    n = G1.shape[0]

    if initialization == 'spatial':
        #spatial_adj = SC_temple_Schaefer.copy()
        spatial_adj = SC_temple_Gla.copy()
        spatial_adj = set_one_in_zero_rows(spatial_adj)
        P0 = spatial_adj / spatial_adj.sum(axis=1)[:, np.newaxis]
        #P0 = spatial_adj
        #P0 = SC_temple_Schaefer
    elif initialization == 'identity':
        P0 = np.eye(n)
    elif initialization == 'barycenter':
        P0 = np.ones((n, n))/n
    else:  # random
        P0 = np.random.rand(n, n)

    # P_aligned = faq_matching(G1, G2, P0)

    # matcher = FAQGraphMatcher(P0, G1, G2)
    # P_aligned = matcher.match()
    from scipy import optimize
    #P0 = sinkhorn_knopp(P0,n)
    P0 =  doubly_stochastic(P0)

    options = {'P0': P0,'tol':0.03}
    P_index = optimize.quadratic_assignment(G1, G2, options=options)

    # options = {"partial_guess": np.array([np.arange(n), P_index.col_ind]).T}
    #
    # P_index = optimize.quadratic_assignment(G1, G2, method="2opt", options=options)

    P_aligned = generate_adjacency_matrix(P_index.col_ind)

    return P_index.col_ind, P_aligned


#%%
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

#%%
# SC1 = np.load("/Users/sonmjack/Downloads/LOGML/brain_connectomes/derivatives/sub-100206/connectome/sub-100206_ses-01_atlas-Schaefer1000_SC.npy")
# SC2 = np.load("/Users/sonmjack/Downloads/LOGML/brain_connectomes/derivatives/sub-101107/connectome/sub-101107_ses-01_atlas-Schaefer1000_SC.npy")
P_matrix_list = []
P_index_list = []
alignment_quality_list = []
permutation_rates_list = []

outpre_path = path_dir("/Users/sonmjack/Downloads/LOGML/All SC Atlas_Gla360")
SC_mean = np.zeros((358,358))
for i in range(len(outpre_path)):
    SC_mean = SC_mean+np.load(outpre_path[i])

SC_mean = SC_mean/len(outpre_path)
SC2 = np.load(outpre_path[0])
#%%
for i in range(len(outpre_path)):
    SC1 = np.load(outpre_path[i])
    n = SC1.shape[0]  # number of brain regions
    G1 = SC1
    G2 = SC_mean
    #spatial_adj = SC_temple_Schaefer
    spatial_adj = SC_temple_Gla.copy()
    spatial_adj = set_one_in_zero_rows(spatial_adj)

    P_index, P_aligned = graph_alignment(G1, G2, initialization='spatial')
    alignment_quality = compute_alignment_quality(G1, G2, P_aligned)

    print(f"Alignment quality: {alignment_quality}")
    alignment_quality_list.append(alignment_quality)
    # Analyze permutations
    permutation_rates = {
        'self': np.diag(P_aligned),
        'neighbors': np.sum(P_aligned * spatial_adj, axis=1) - np.diag(P_aligned),
        'others': np.sum(P_aligned, axis=1) - np.sum(P_aligned * spatial_adj, axis=1)
    }

    permutation_types = analyze_permutations(P_aligned, spatial_adj)

    permutation_rates_list.append(permutation_types)

    print("Permutation types:")
    for perm_type, ratio in permutation_types.items():
        print(f"{perm_type}: {ratio:.2%}")

    P_matrix_list.append(P_aligned)
    P_index_list.append(P_index)

    print(f'Finish_{i}')

#%%
import pickle
with open('/Users/sonmjack/Downloads/LOGML/P_matrix_list.pkl', 'wb') as file:
    pickle.dump(P_matrix_list, file)
with open('/Users/sonmjack/Downloads/LOGML/P_index_list.pkl', 'wb') as file:
    pickle.dump(P_index_list, file)

with open('/Users/sonmjack/Downloads/LOGML/permutation_rates_list.pkl', 'wb') as file:
    pickle.dump(permutation_rates_list, file)
with open('/Users/sonmjack/Downloads/LOGML/alignment_quality_list.pkl', 'wb') as file:
    pickle.dump(alignment_quality_list, file)