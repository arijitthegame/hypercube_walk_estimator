import numpy as np
from math import comb

def trace_curie_weiss_diagonal_only(n, beta, J):
    """Exact trace of exp(-beta * diagonal CW part) — no transverse field."""
    return sum(
        comb(n, k) * np.exp(beta * J * (n - 2*k)**2 / n)
        for k in range(n+1)
    )

def trace_1D_TFIM_ring(n, beta, J, Gamma):
    """
    Exact via Jordan-Wigner. Only valid when periodic boundary conditions.
    Quasi-particle energies: eps_k = sqrt((J*cos(2pi*k/n))^2 + Gamma^2)
    """
    k = np.arange(n)
    eps_k = np.sqrt((J * np.cos(2 * np.pi * k / n))**2 + Gamma**2)
    return np.prod(2 * np.cosh(beta * eps_k))

def trace_hypercube(n, beta, J):
    """2^n * cosh(beta*J)^n"""
    return (2**n) * np.cosh(beta * J)**n


def trace_complete(N, beta, J):
    """e^{beta*J*(N-1)} + (N-1)*e^{-beta*J}"""
    return np.exp(beta * J * (N-1)) + (N-1) * np.exp(-beta * J)

def trace_complete_bipartite(N, beta, J):
    """2*cosh(beta*J*N) + 2*(N-1)"""
    return 2 * np.cosh(beta * J * N) + 2 * (N - 1)

def trace_ring(N, beta, J):
    """Sum over eigenvalues 2*cos(2*pi*k/N)"""
    k = np.arange(N)
    eigenvalues = 2 * np.cos(2 * np.pi * k / N)
    return np.sum(np.exp(beta * J * eigenvalues))


def trace_U_hypercube(n, sigma):
    """degree = n"""
    return (2**n) * np.cosh(sigma**2 / n)**n

def trace_U_complete(N, sigma):
    """degree = N-1"""
    return np.exp(sigma**2) + (N-1) * np.exp(-sigma**2 / (N-1))

def trace_U_complete_bipartite(N, sigma):
    """degree = N, the N cancels cleanly"""
    return 2 * np.cosh(sigma**2) + 2 * (N - 1)

def trace_U_ring(N, sigma):
    """degree = 2"""
    k = np.arange(N)
    eigenvalues = 2 * np.cos(2 * np.pi * k / N)
    return np.sum(np.exp((sigma**2 / 2) * eigenvalues))
