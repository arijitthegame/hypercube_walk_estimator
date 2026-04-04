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



