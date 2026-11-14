## Stand alone file to understand the structure of Hamiltonians
## USeful for people without a physics background.

import numpy as np
from math import comb

def adjacency_hypercube(n):
    """Adjacency matrix M_n of the n-dimensional hypercube."""
    if n == 0:
        return np.array([[0]])
    M1 = np.array([[0, 1], [1, 0]])
    M = M1.copy()
    for _ in range(n - 1):
        I_prev = np.eye(M.shape[0], dtype=int)
        I2     = np.eye(2,         dtype=int)
        M = np.kron(M, I2) + np.kron(I_prev, M1)
    return M


def adjacency_complete(N):
    """K_N: complete graph on N vertices."""
    return np.ones((N, N)) - np.eye(N)

def adjacency_complete_bipartite(N):
    """K_{N,N}: complete bipartite graph, 2N vertices."""
    A = np.zeros((2*N, 2*N))
    A[:N, N:] = 1
    A[N:, :N] = 1
    return A

def adjacency_ring(N):
    """C_N: cycle/ring on N vertices."""
    A = np.zeros((N, N))
    for i in range(N):
        A[i, (i+1) % N] = 1
        A[i, (i-1) % N] = 1
    return A

def adjacency_TFIM_curie_weiss(n):
    """
    Diagonal part of Curie-Weiss quantum Ising.
    Diagonal entries are -(sum of sigma_z)^2 / n for each config.
    Returns the full 2^n x 2^n diagonal matrix.
    """
    size = 1 << n
    A = np.zeros((size, size))
    for x in range(size):
        # magnetization = n - 2*(number of 1-bits)
        mag = n - 2 * bin(x).count('1')
        A[x, x] = (mag ** 2) / n
    return A

def hamiltonian_TFIM_curie_weiss(n, J, Gamma):
    """Full Curie-Weiss TFIM Hamiltonian."""
    return -J * adjacency_TFIM_curie_weiss(n) - Gamma * adjacency_hypercube(n)

def hamiltonian_TFIM_1D_ring(n, J, Gamma):
    """
    1D TFIM on a ring.
    Diagonal: -J * sigma_z_i * sigma_z_{i+1} (classical Ising on ring)
    Off-diagonal: -Gamma * M_n (transverse field = hypercube)
    """
    size = 1 << n
    diag = np.zeros(size)
    for x in range(size):
        energy = 0
        for i in range(n):
            si = 1 - 2*((x >> i) & 1)       # +1 or -1
            sj = 1 - 2*((x >> ((i+1)%n)) & 1)
            energy += si * sj
        diag[x] = -J * energy
    return np.diag(diag) - Gamma * adjacency_hypercube(n)

#################################################################
### General Transverse Ising Models #######################

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=np.float64)
X = np.array([[0, 1], [1, 0]], dtype=np.float64)
Z = np.array([[1, 0], [0, -1]], dtype=np.float64)


def kron_n(ops):
    """Kronecker product of a list of matrices."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def local_operator(n, site, op):
    """
    Construct operator acting on 'site' (0-indexed) in n-spin system.
    """
    ops = [I] * n
    ops[site] = op
    return kron_n(ops)


def two_site_operator(n, i, j, op_i, op_j):
    """
    Construct operator acting on sites i and j.
    """
    ops = [I] * n
    ops[i] = op_i
    ops[j] = op_j
    return kron_n(ops)


def transverse_ising_hamiltonian(n, J, h):
    """
    Construct H = - sum_{i<j} J[i,j] Z_i Z_j - sum_i h[i] X_i

    Parameters:
    - n: number of spins
    - J: (n x n) symmetric coupling matrix
    - h: (n,) transverse field vector
    """
    dim = 2**n
    H = np.zeros((dim, dim), dtype=np.float64)

    # Interaction term: Z_i Z_j
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0:
                H -= J[i, j] * two_site_operator(n, i, j, Z, Z)

    # Transverse field: X_i
    for i in range(n):
        if h[i] != 0:
            H -= h[i] * local_operator(n, i, X)

    return H






