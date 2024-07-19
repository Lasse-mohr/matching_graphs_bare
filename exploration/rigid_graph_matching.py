import numpy as np
import scipy.sparse as sp
import os
import sys

# Add the base directory to the system path for imports
base_dir = os.path.abspath('..')
sys.path.append(base_dir)

from graph_matching.data_loader.graph_loader import NetworkLoader, compute_mean_network
from graph_matching.netalign.create_matrices import LMatrixCreater

def bound(S, a, b):
    S_plus_b = S.copy()
    S_plus_b.data += b
    S_plus_a = S.copy()
    S_plus_a.data += a
    return sp.csr_matrix(np.maximum(S_plus_b.data, 0) - np.maximum(S_plus_a.data, 0))

def maxprod(ai, aj, av, m, x):
    y = -np.inf * np.ones(m)
    for i in range(len(ai)):
        y[ai[i]] = max(y[ai[i]], av[i] * x[aj[i]])
    return y

def implicit_maxprod(n, ai, x):
    N = len(ai)
    y = -np.inf * np.ones(N)
    max1 = -np.inf * np.ones(n)
    max2 = -np.inf * np.ones(n)
    max1ind = np.zeros(n, dtype=int)
    for i in range(N):
        if x[i] > max2[ai[i]]:
            if x[i] > max1[ai[i]]:
                max2[ai[i]] = max1[ai[i]]
                max1[ai[i]] = x[i]
                max1ind[ai[i]] = i
            else:
                max2[ai[i]] = x[i]
    for i in range(N):
        if i == max1ind[ai[i]]:
            y[i] = max2[ai[i]]
        else:
            y[i] = max1[ai[i]]
    return y

def round_messages(messages, S, w, alpha, beta, rp, ci, tripi, n, m, perm):
    ai = np.zeros(len(tripi))
    ai[tripi > 0] = messages[perm]
    val, ma, mb, mi = bipartite_matching_primal_dual(rp, ci, ai, tripi, n, m)
    matchweight = np.sum(w[mi])
    cardinality = np.sum(mi)
    overlap = (mi @ (S @ mi.astype(np.float64))) / 2
    f = alpha * matchweight + beta * overlap
    return np.array([f, matchweight, cardinality, overlap])

def bipartite_matching_setup(w, li, lj, m, n):
    nedges = len(w)
    rp = np.zeros(m + 1, dtype=int)
    ci = np.zeros(nedges + n, dtype=int)
    ai = np.zeros(nedges + n)
    tripi = np.zeros(nedges + n, dtype=int)

    rp[0] = 0
    for i in range(nedges):
        if li[i] + 1 >= len(rp):
            print(f"Error: index {li[i] + 1} out of bounds for rp with length {len(rp)}")
        rp[li[i] + 1] += 1

    rp = np.cumsum(rp)
    rp_copy = np.copy(rp)

    for i in range(nedges):
        if li[i] >= len(rp_copy) or rp_copy[li[i]] >= len(tripi):
            print(f"Error: index {li[i]} or {rp_copy[li[i]]} out of bounds in tripi assignment")
        tripi[rp_copy[li[i]]] = i + 1
        ai[rp_copy[li[i]]] = w[i]
        ci[rp_copy[li[i]]] = lj[i]
        rp_copy[li[i]] += 1

    for i in range(n):
        if rp_copy[i] >= len(tripi):
            print(f"Error: index {rp_copy[i]} out of bounds in tripi assignment")
        tripi[rp_copy[i]] = -1
        ai[rp_copy[i]] = 0
        ci[rp_copy[i]] = m + i
        rp_copy[i] += 1

    for i in range(n, 0, -1):
        rp[i] = rp[i - 1]
    rp[0] = 0
    rp = rp + 1

    colind = np.zeros(m + n, dtype=bool)
    for i in range(n):
        for rpi in range(rp[i] - 1, rp[i + 1] - 1):
            if ci[rpi] >= len(colind):
                print(f"Error: index {ci[rpi]} out of bounds in colind")
            if colind[ci[rpi]]:
                raise ValueError(f"Duplicate edge detected ({i}, {ci[rpi]})")
            colind[ci[rpi]] = True
        for rpi in range(rp[i] - 1, rp[i + 1] - 1):
            colind[ci[rpi]] = False

    return rp, ci, ai, tripi, n, m

def bipartite_matching_primal_dual(rp, ci, ai, tripi, n, m):
    alpha = np.zeros(n)
    beta = np.zeros(n + m)
    queue = np.zeros(n + m, dtype=int)
    t = np.zeros(n + m, dtype=int)
    match1 = np.zeros(n, dtype=int)
    match2 = np.zeros(n + m, dtype=int)
    tmod = np.zeros(n + m, dtype=int)
    ntmod = 0

    for i in range(n):
        for rpi in range(rp[i] - 1, rp[i + 1] - 1):
            if ai[rpi] > alpha[i]:
                alpha[i] = ai[rpi]

    i = 0
    while i < n:
        for j in range(ntmod):
            t[tmod[j]] = 0
        ntmod = 0

        head = 0
        tail = 0
        queue[head] = i + 1
        head += 1

        while head > tail and match1[i] == 0:
            k = queue[tail] - 1
            tail += 1
            for rpi in range(rp[k] - 1, rp[k + 1] - rp[k]):
                j = ci[rpi]
                if j >= len(beta) or ai[rpi] < alpha[k] + beta[j] - 1e-8:
                    continue
                if t[j] == 0:
                    queue[head] = match2[j]
                    head += 1
                    t[j] = k + 1
                    tmod[ntmod] = j
                    ntmod += 1
                    if match2[j] == 0:
                        while j > 0:
                            match2[j] = t[j]
                            k = t[j] - 1
                            temp = match1[k]
                            match1[k] = j
                            j = temp
                        break

        if match1[i] == 0:
            theta = np.inf
            for j in range(tail):
                t1 = queue[j] - 1
                for rpi in range(rp[t1] - 1, rp[t1 + 1] - rp[t1]):
                    t2 = ci[rpi]
                    if t2 >= len(beta) or t[t2] != 0:
                        continue
                    if alpha[t1] + beta[t2] - ai[rpi] < theta:
                        theta = alpha[t1] + beta[t2] - ai[rpi]

            for j in range(tail):
                alpha[queue[j] - 1] -= theta

            for j in range(ntmod):
                beta[tmod[j]] += theta

            continue

        i += 1

    val = 0
    for i in range(n):
        for rpi in range(rp[i] - 1, rp[i + 1] - rp[i]):
            if ci[rpi] == match1[i]:
                val += ai[rpi]

    noute = np.sum(match1[:n] <= m)
    m1 = np.zeros(noute, dtype=int)
    m2 = np.zeros(noute, dtype=int)
    noute = 0
    for i in range(n):
        if match1[i] <= m:
            m1[noute] = i + 1
            m2[noute] = match1[i]
            noute += 1

    if tripi is not None:
        mi = np.zeros(len(tripi) - n, dtype=bool)
        for i in range(n):
            for rpi in range(rp[i] - 1, rp[i + 1] - rp[i]):
                if match1[i] <= m and ci[rpi] == match1[i]:
                    mi[tripi[rpi] - 1] = True

        return val, m1, m2, mi

    return val, m1, m2


def netalignmbp(S, w, a=1, b=1, li=None, lj=None, gamma=0.99, dtype=2, maxiter=100, verbose=1):
    nedges = len(li)
    nsquares = S.nnz // 2
    m = max(li) + 1
    n = max(lj) + 1

    if len(w) != len(li):
        raise ValueError("Length of w does not match length of li")

    y = np.zeros(nedges)
    z = np.zeros(nedges)
    Sk = sp.csr_matrix(S.shape)
    if dtype > 1:
        d = np.zeros(nedges)

    damping = gamma
    curdamp = 1
    iter = 1

    hista = np.zeros((maxiter, 4))
    histb = np.zeros((maxiter, 4))
    fbest = 0
    fbestiter = 0

    if verbose:
        print(f'{"best":<4} {"iter":<4} {"obj_ma":<7} {"wght_ma":<7} {"card_ma":<7} {"over_ma":<7} {"obj_mb":<7} {"wght_mb":<7} {"card_mb":<7} {"over_mb":<7}')

    rp, ci, ai, tripi, matn, matm = bipartite_matching_setup(w, li, lj, m, n)
    mperm = tripi[tripi > 0]

    while iter <= maxiter:
        curdamp = damping * curdamp
        Sknew = bound(Sk.transpose() + b * S, 0, b)
        
        if dtype > 1:
            dold = d
        
        d = np.sum(Sknew, axis=1).A1

        ynew = a * w - np.maximum(0, implicit_maxprod(n, lj, z)) + d
        znew = a * w - np.maximum(0, implicit_maxprod(m, li, y)) + d

        diag_elements = ynew + znew - a * w - d
        diag_matrix = sp.diags(diag_elements)
        
        print("Iteration:", iter)
        print("Shape of diagonal matrix:", diag_matrix.shape)
        print("Shape of S:", S.shape)
        print("Shape of Sknew:", Sknew.shape)
        
        Skt = diag_matrix @ S
        print("Shape after multiplication:", Skt.shape)
        
        assert Skt.shape == Sknew.shape, f"Shapes of Skt ({Skt.shape}) and Sknew ({Sknew.shape}) must be the same for subtraction"

        Skt = Skt - Sknew

        if dtype == 1:
            Sk = curdamp * Skt + (1 - curdamp) * Sk
            y = curdamp * ynew + (1 - curdamp) * y
            z = curdamp * znew + (1 - curdamp) * z
        elif dtype == 2:
            prev = y + z - a * w + dold
            y = ynew + (1 - curdamp) * prev
            z = znew + (1 - curdamp) * prev
            Sk = Skt + (1 - curdamp) * (Sk + Sk.transpose() - b * S)
        elif dtype == 3:
            prev = y + z - a * w + dold
            y = curdamp * ynew + (1 - curdamp) * prev
            z = curdamp * znew + (1 - curdamp) * prev
            Sk = curdamp * Skt + (1 - curdamp) * (Sk + Sk.transpose() - b * S)

        hista[iter - 1] = round_messages(y, S, w, a, b, rp, ci, tripi, matn, matm, mperm)
        histb[iter - 1] = round_messages(z, S, w, a, b, rp, ci, tripi, matn, matm, mperm)

        if hista[iter - 1, 0] > fbest:
            fbestiter = iter
            mbest = y
            fbest = hista[iter - 1, 0]

        if histb[iter - 1, 0] > fbest:
            fbestiter = -iter
            mbest = z
            fbest = histb[iter - 1, 0]

        if verbose:
            bestchar = '*a' if fbestiter == iter else '*b' if fbestiter == -iter else ''
            print(f'{bestchar:<4} {iter:<4} {hista[iter - 1, 0]:<7g} {hista[iter - 1, 1]:<7g} {hista[iter - 1, 2]:<7g} {hista[iter - 1, 3]:<7g} {histb[iter - 1, 0]:<7g} {histb[iter - 1, 1]:<7g} {histb[iter - 1, 2]:<7g} {histb[iter - 1, 3]:<7g}')

        iter += 1

    return S

def sparse_to_csr_components(matrix):
    csr = sp.csr_matrix(matrix)
    rp = csr.indptr
    ci = csr.indices
    v = csr.data
    return rp, ci, v

def make_squares(A, B, L):
    n = A.shape[0]
    m = B.shape[0]

    rpA, ciA, _ = sparse_to_csr_components(A)
    rpB, ciB, _ = sparse_to_csr_components(B)
    rpAB, ciAB, vAB = sparse_to_csr_components(L)

    Se = np.zeros((1, 2), dtype=int)
    wv = np.zeros(m, dtype=int)
    sqi = 0

    for i in range(n):
        for ri1 in range(rpAB[i], rpAB[i+1]):
            wv[ciAB[ri1]] = ri1
        
        for ri1 in range(rpA[i], rpA[i+1]):
            ip = ciA[ri1]
            if i == ip:
                continue
            for ri2 in range(rpAB[ip], rpAB[ip+1]):
                jp = ciAB[ri2]
                for ri3 in range(rpB[jp], rpB[jp+1]):
                    j = ciB[ri3]
                    if j == jp:
                        continue
                    if wv[j] > 0:
                        sqi += 1
                        if sqi > Se.shape[0]:
                            Se = np.vstack((Se, Se))

                        Se[sqi-1, 0] = ri2
                        Se[sqi-1, 1] = wv[j]

    Se = Se[:sqi, :]

    if n > 1:
        Le = np.zeros((len(vAB), 3), dtype=int)
        for i in range(n):
            for j in range(rpAB[i], rpAB[i+1]):
                Le[j, 0] = i
                Le[j, 1] = ciAB[j]
                Le[j, 2] = vAB[j]
    else:
        Le = None

    return Se, Le

def construct_L(matrix_A, matrix_B):
        # Initialize the output matrix L with appropriate dimensions
        L = np.zeros((matrix_A.shape[0], matrix_B.shape[1]))

        # Calculate all distances and find the minimum distance
        distances = np.zeros((matrix_A.shape[0], matrix_B.shape[1]))
        for row in range(matrix_A.shape[0]):
            for col in range(matrix_B.shape[1]):
                distance = np.linalg.norm(matrix_A[row,:] - matrix_B[:, col], ord=2)**2
                distances[row, col] = distance
        
        # Find the minimum distance
        max_distance = np.max(distances)

        # Normalize distances and populate L
        for row in range(matrix_A.shape[0]):
            for col in range(matrix_B.shape[1]):
                normalized_distance = distances[row, col] / max_distance
                print(normalized_distance)  
                if normalized_distance < 1e-4: 
                    L[row, col] = 1

        return L

def compute(CA, CB):
    L = construct_L(CA, CB)
    A = (CA != 0).astype(int)
    B = (CB != 0).astype(int)
    Se, Le = make_squares(A, B, L)
    li = Le[:,0]
    lj = Le[:, 1]
    w = Le[:,2]
    S = sp.coo_matrix((np.ones(len(Se[:, 0])), (Se[:, 0], Se[:, 1])), shape=(np.count_nonzero(L), np.count_nonzero(L)))
    X = netalignmbp(S, w, 1, 1, li, lj)
    B = X @ B @ X.T

    AtB = np.dot(CA.T, CB)
    U, S, Vt = np.linalg.svd(AtB)
    R = np.dot(U, Vt)
    rotation_angle = np.degrees(np.arccos(R[0, 0]))
    if R[1, 0] < 0:
        rotation_angle = -rotation_angle
    CB_rotated = np.dot(CB, R.T)
    return CA, CB_rotated

def construct_L(matrix_A, matrix_B):
    L = np.zeros((matrix_A.shape[0], matrix_B.shape[1]))
    distances = np.zeros((matrix_A.shape[0], matrix_B.shape[1]))
    for row in range(matrix_A.shape[0]):
        for col in range(matrix_B.shape[1]):
            distance = np.linalg.norm(matrix_A[row,:] - matrix_B[:, col], ord=2)**2
            distances[row, col] = distance
    
    max_distance = np.max(distances)

    for row in range(matrix_A.shape[0]):
        for col in range(matrix_B.shape[1]):
            normalized_distance = distances[row, col] / max_distance
            if normalized_distance < 1e-4:
                L[row, col] = 1

    return L

if __name__ == "__main__":
    atlas = 'Schaefer1000'
    structure_type = 'ses-01'
    loader = NetworkLoader(atlas, structure_type, base_dir=base_dir+'\\matching_graphs_spatial_constraints-main\\data_new_struct')

    _, network = next(loader)
    mean_network = compute_mean_network(loader)

    CA = network
    CB = mean_network
    L = construct_L(network, mean_network)
    compute(CA, CB)
