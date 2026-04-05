"""
spin_trace_estimator.py
=======================
Graph Random Feature (GRF) + Hutchinson trace estimator for spin system
Hamiltonians of the form:

    H = D + M

where:
    D  = diagonal matrix of ZZ (or p-body) coupling energies
    M  = off-diagonal matrix of transverse field (hypercube edges)

TWO ESTIMATION MODES
--------------------
This module supports two modes, controlled by the `mode` argument in
`trace_estimate()`:

    mode='M_only'  (default)
    -------------------------
    Estimates tr(f(M)), i.e. the partition function of the PURE transverse
    field with no ZZ coupling. This estimator is EXACTLY UNBIASED for
    tr(f(M)).

    When is this useful?
    - J = 0 (no ZZ coupling), in which case H = M exactly and the
      estimator is unbiased for tr(f(H)).
    - High temperature (small beta) or weak coupling (small ||J||),
      where tr(f(M)) approximates tr(f(H)) well. The bias is
      O(beta^2 ||J||_F ||h||_2) to leading order (see latex doc).
    - As a control variate: since tr(e^{-beta M}) has a closed form
      for the hypercube (2^n * cosh(beta||h||/n)^n), the difference
      tr(f(H)) - tr(f(M)) can be estimated separately and added back.

    Ground truth to compare against:
        H_offdiag = system.build_hamiltonian()
        np.fill_diagonal(H_offdiag, 0)
        exact_M = np.trace(expm(-beta * H_offdiag))

    mode='trotter'
    --------------
    Estimates tr(e^{-beta D} e^{-beta M}) via a symmetric Trotter
    (Strang) splitting:

        e^{-beta H} = e^{-beta D/2} e^{-beta M} e^{-beta D/2} + O(beta^3)

    The diagonal correction e^{-beta D_{xx}/2} is absorbed as a
    multiplicative weight at the start (left factor) and at each visited
    node (right factor) during the walk. This estimator targets
    tr(e^{-beta D} e^{-beta M}), which approximates tr(e^{-beta H})
    with bias O(beta^3 ||[D,M]||^2), one order better than mode='M_only'.

    NOTE: This mode only makes sense for f(H) = e^{-beta H}. For other
    kernel functions (resolvent, p-step RW), use mode='M_only' and
    interpret the result accordingly.

    Ground truth to compare against:
        H = system.build_hamiltonian()
        D = np.diag(np.diag(H))
        M = H - D
        exact_trotter = np.trace(expm(-beta * D) @ expm(-beta * M))

BIAS SUMMARY
------------
    Mode        Targets                     Bias vs tr(f(H))
    --------    -------------------------   ----------------------------
    M_only      tr(f(M))                    O(beta^2 ||J||_F ||h||_2)
    trotter     tr(e^{-bD} e^{-bM})         O(beta^3 ||[D,M]||^2)
    exact        tr(f(H))                   0  (only via expm, O(8^n))

HOW TO TEST WHICH VERSION IS WHICH
------------------------------------
Run the built-in `verify()` method on any system. It prints a table
comparing the estimator against three ground truths:

    - exact_M      : np.trace(expm(-beta * M))     <- unbiased target of M_only
    - exact_trotter: np.trace(expm(-beta*D) @ expm(-beta*M))  <- unbiased target of trotter
    - exact_H      : np.trace(expm(-beta * H))     <- true partition function

A well-functioning estimator should have near-zero error against its
own target, and nonzero error against the others (unless J ~ 0).

COMPLEXITY
----------
    Per walk step:
        mode='M_only'  : O(n)       neighbors + weights from h only
        mode='trotter' : O(n*d)     same walk + diagonal_entry at each node
                         where d = average degree of J coupling graph

    Total:
        O(S * R * L * n)  for M_only
        O(S * R * L * n*d) for trotter
    where S = nb_hutchinson, R = nb_walks, L = mean walk length = (1-p)/p

USAGE EXAMPLE
-------------
    import numpy as np
    from spin_trace_estimator import TFIM, f_diffusion

    n, beta = 4, 0.5
    J = np.zeros((n, n))
    for i in range(n-1):
        J[i, i+1] = J[i+1, i] = 0.4
    h = np.full(n, 0.3)

    system = TFIM(n, J, h)

    # Unbiased for tr(e^{-beta M})
    est_M = system.trace_estimate(
        f=f_diffusion(beta), beta=beta,
        p_halt=0.3, nb_walks=64, nb_hutchinson=200,
        mode='M_only'
    )

    # Trotter-corrected approximation of tr(e^{-beta H})
    est_H = system.trace_estimate(
        f=f_diffusion(beta), beta=beta,
        p_halt=0.3, nb_walks=64, nb_hutchinson=200,
        mode='trotter'
    )

    # Verify both with ground truth table
    system.verify(beta=beta, p_halt=0.3, nb_walks=64, nb_hutchinson=200)
"""

import numpy as np
import scipy
from scipy.linalg import expm
from itertools import combinations


# ═══════════════════════════════════════════════════════════════════
# BASE CLASS
# ═══════════════════════════════════════════════════════════════════

class SpinSystem:
    """
    Abstract base class for spin Hamiltonians H = D + M.

    Subclasses must implement:
        get_neighbors_and_weights(x) -> (neighbors, weights)
        diagonal_entry(x)            -> float

    Everything else (walk, GRF, Hutchinson, verify) is inherited.
    """

    # ── Interface subclasses must implement ───────────────────────

    def get_neighbors_and_weights(self, x):
        """
        Off-diagonal neighbors of configuration x and their H matrix weights.

        Parameters
        ----------
        x : int
            Spin configuration encoded as a bitmask.

        Returns
        -------
        neighbors : list of int
            Neighboring configurations.
        weights : np.ndarray
            Signed off-diagonal H entries H[x, neighbor].
            These are the RAW matrix entries, not probabilities.
        """
        raise NotImplementedError

    def diagonal_entry(self, x):
        """
        Diagonal entry H[x, x] for configuration x.

        Parameters
        ----------
        x : int
            Spin configuration encoded as a bitmask.

        Returns
        -------
        float
            H[x, x], the on-site energy of configuration x.
        """
        raise NotImplementedError

    # ── Shared internals ──────────────────────────────────────────

    def _transition_probs(self, weights):
        """Transition probabilities proportional to |weights|."""
        abs_w = np.abs(weights)
        total = abs_w.sum()
        if total == 0:
            return np.zeros_like(abs_w)
        return abs_w / total

    def _random_walk(self, start, p_halt):
        """
        Single random walk from `start` halting with prob p_halt at each step.

        Returns
        -------
        list of (vertex, neighbors, weights, probs)
            Full walk history including all information needed for
            importance weight computation.
        """
        walk = []
        x    = start
        while True:
            neighbors, weights = self.get_neighbors_and_weights(x)
            weights = np.array(weights, dtype=float)
            probs   = self._transition_probs(weights)
            walk.append((x, neighbors, weights, probs))

            if np.random.uniform() < p_halt or len(neighbors) == 0:
                break

            idx = np.random.choice(len(neighbors), p=probs)
            x   = neighbors[idx]
        return walk

    def _grf_vector_M_only(self, v, p_halt, nb_walks, f_vec):
        """
        Estimate column v of f(M) via random walks on the off-diagonal
        graph only. Diagonal D is completely ignored.

        Unbiased for [f(M)]_{:, v}.

        Returns sparse dict {node: estimated f(M)[node, v]}.
        """
        rf = {}
        for _ in range(nb_walks):
            walk            = self._random_walk(v, p_halt)
            weights_product = 1.0
            marginal_prob   = 1.0

            for step, (node, neighbors, weights, probs) in enumerate(walk):
                if step < len(f_vec):
                    contribution = (weights_product / marginal_prob) * f_vec[step]
                    rf[node]     = rf.get(node, 0.0) + contribution

                if step < len(walk) - 1:
                    next_node        = walk[step + 1][0]
                    idx              = list(neighbors).index(next_node)
                    weights_product *= weights[idx]
                    marginal_prob   *= (1 - p_halt) * probs[idx]

        return {node: val / nb_walks for node, val in rf.items()}

    def _grf_vector_trotter(self, v, p_halt, nb_walks, f_vec, beta):
        """
        Estimate column v of e^{-beta D/2} f(M) e^{-beta D/2} via the
        symmetric Trotter correction.

        The left factor e^{-beta D[v,v]/2} is applied at the start vertex.
        The right factor e^{-beta D[node,node]/2} is applied at each
        visited node. Together they approximate the full e^{-beta H}
        entry up to O(beta^3) Trotter error.

        Unbiased for [e^{-beta D} e^{-beta M}]_{:, v} up to Trotter error.

        Returns sparse dict {node: estimated value}.
        """
        rf          = {}
        left_factor = np.exp(-beta * self.diagonal_entry(v) / 2.0)

        for _ in range(nb_walks):
            walk            = self._random_walk(v, p_halt)
            # Absorb left Trotter factor into starting weight
            weights_product = left_factor
            marginal_prob   = 1.0

            for step, (node, neighbors, weights, probs) in enumerate(walk):
                if step < len(f_vec):
                    # Absorb right Trotter factor at current node
                    right_factor = np.exp(-beta * self.diagonal_entry(node) / 2.0)
                    contribution = (weights_product / marginal_prob) * f_vec[step] * right_factor
                    rf[node]     = rf.get(node, 0.0) + contribution

                if step < len(walk) - 1:
                    next_node        = walk[step + 1][0]
                    idx              = list(neighbors).index(next_node)
                    weights_product *= weights[idx]
                    marginal_prob   *= (1 - p_halt) * probs[idx]

        return {node: val / nb_walks for node, val in rf.items()}

    # ── Public API ────────────────────────────────────────────────

    def trace_estimate(self, f, beta, p_halt, nb_walks, nb_hutchinson,
                       mode='M_only', max_walk_length=50, verbose=False):
        """
        Estimate tr(f(H)) via Hutchinson + GRF.

        Parameters
        ----------
        f : callable
            Modulation function f(k) -> float. Taylor coefficient of f
            at order k. For e^{-beta H} use f_diffusion(beta).
        beta : float
            Inverse temperature. Used only in mode='trotter' for the
            diagonal correction. Ignored in mode='M_only'.
        p_halt : float
            Walk halting probability at each step. Mean walk length = (1-p)/p.
            Smaller p_halt -> longer walks -> better approximation of f
            but higher variance.
        nb_walks : int
            Number of random walks per vertex per Hutchinson sample.
        nb_hutchinson : int
            Number of Hutchinson probe vectors. More -> lower variance.
        mode : str, optional
            'M_only'  : estimate tr(f(M)), unbiased, ignores D (default).
            'trotter' : estimate tr(e^{-bD} e^{-bM}), O(beta^3) bias.
        max_walk_length : int, optional
            Precompute f(k) up to this depth. Should exceed typical walk
            length 1/p_halt by a comfortable margin.
        verbose : bool, optional
            Print per-sample estimates and running mean.

        Returns
        -------
        float
            Trace estimate.

        Notes
        -----
        The estimator is unbiased for:
            mode='M_only'  -> tr(f(M))
            mode='trotter' -> tr(e^{-beta D} e^{-beta M})
        Neither is unbiased for tr(f(H)) when J != 0.
        See module docstring for bias analysis.
        """
        assert mode in ('M_only', 'trotter'), \
            "mode must be 'M_only' or 'trotter'"

        size  = 1 << self.n
        f_vec = [f(k) for k in range(max_walk_length)]

        # Select GRF vector function based on mode
        if mode == 'M_only':
            grf_fn = lambda v: self._grf_vector_M_only(
                v, p_halt, nb_walks, f_vec)
        else:
            grf_fn = lambda v: self._grf_vector_trotter(
                v, p_halt, nb_walks, f_vec, beta)

        estimates = []
        for s in range(nb_hutchinson):
            z   = {x: np.random.choice([-1.0, 1.0]) for x in range(size)}
            fHz = {}

            for v in range(size):
                rf = grf_fn(v)
                for node, val in rf.items():
                    fHz[node] = fHz.get(node, 0.0) + z[v] * val

            est = sum(z[node] * val for node, val in fHz.items())
            estimates.append(est)
            if verbose:
                print(f"  [{mode}] Sample {s+1:3d}: {est:.4f}  "
                      f"running mean = {np.mean(estimates):.4f}")

        return float(np.mean(estimates))

    def build_hamiltonian(self):
        """
        Build the full 2^n x 2^n Hamiltonian matrix H = D + M.
        Only for small n verification. Cost O(4^n).
        """
        size = 1 << self.n
        H    = np.zeros((size, size))
        for x in range(size):
            H[x, x] = self.diagonal_entry(x)
            neighbors, weights = self.get_neighbors_and_weights(x)
            for nb, w in zip(neighbors, weights):
                H[x, nb] = w
        return H

    def verify(self, beta, p_halt, nb_walks, nb_hutchinson,
               max_walk_length=50):
        """
        Run both estimator modes and compare against three ground truths.

        Ground truths
        -------------
        exact_M       : tr(e^{-beta M})
                        Unbiased target of mode='M_only'.
        exact_trotter : tr(e^{-beta D} e^{-beta M})
                        Unbiased target of mode='trotter'.
        exact_H       : tr(e^{-beta H})
                        True partition function. Neither mode is unbiased
                        for this when J != 0.

        A correct implementation should show:
            mode='M_only'  has ~0 error vs exact_M,  nonzero vs exact_H
            mode='trotter' has ~0 error vs exact_trotter, smaller vs exact_H
        """
        H = self.build_hamiltonian()
        D = np.diag(np.diag(H))
        M = H - D

        exact_H       = np.trace(expm(-beta * H))
        exact_M       = np.trace(expm(-beta * M))
        exact_trotter = np.trace(expm(-beta * D) @ expm(-beta * M))

        f = f_diffusion(beta)

        est_M_only = self.trace_estimate(
            f=f, beta=beta, p_halt=p_halt, nb_walks=nb_walks,
            nb_hutchinson=nb_hutchinson, mode='M_only',
            max_walk_length=max_walk_length
        )
        est_trotter = self.trace_estimate(
            f=f, beta=beta, p_halt=p_halt, nb_walks=nb_walks,
            nb_hutchinson=nb_hutchinson, mode='trotter',
            max_walk_length=max_walk_length
        )

        print(f"\n{'='*65}")
        print(f"  System : {self.__class__.__name__}  |  "
              f"n={self.n}  beta={beta}  p_halt={p_halt}")
        print(f"{'='*65}")
        print(f"\n  Ground truths:")
        print(f"    exact_M        = {exact_M:.6f}   <- unbiased target of M_only")
        print(f"    exact_trotter  = {exact_trotter:.6f}   <- unbiased target of trotter")
        print(f"    exact_H        = {exact_H:.6f}   <- true partition function")
        print(f"\n  Estimates:")
        print(f"{'Mode':<12} {'Estimate':>10} {'err vs M':>12} "
              f"{'err vs Trotter':>16} {'err vs H':>12}")
        print(f"  {'-'*60}")
        for name, est in [("M_only", est_M_only), ("trotter", est_trotter)]:
            err_M  = abs(est - exact_M)       / abs(exact_M)
            err_T  = abs(est - exact_trotter) / abs(exact_trotter)
            err_H  = abs(est - exact_H)       / abs(exact_H)
            print(f"  {name:<12} {est:>10.4f} {err_M:>11.3%} "
                  f"{err_T:>15.3%} {err_H:>11.3%}")
        print(f"\n  Bias from ignoring D:")
        print(f"    |tr(e^{{-bM}}) - tr(e^{{-bH}})|  = "
              f"{abs(exact_M - exact_H):.6f}")
        print(f"    |tr(e^{{-bD}}e^{{-bM}}) - tr(e^{{-bH}})| = "
              f"{abs(exact_trotter - exact_H):.6f}")
        print(f"  (Trotter reduces bias by factor "
              f"{abs(exact_M-exact_H)/max(abs(exact_trotter-exact_H),1e-12):.1f}x)\n")


# ═══════════════════════════════════════════════════════════════════
# SYSTEM 1: Standard TFIM
# ═══════════════════════════════════════════════════════════════════

class TFIM(SpinSystem):
    """
    Transverse Field Ising Model.

        H = -sum_{i<j} J_ij Z_i Z_j - sum_i h_i X_i

    Off-diagonal part M: single spin flips, weight -h_i.
    Diagonal part D:     two-body ZZ coupling energies.

    Neighbor cost : O(n)     — one bit flip per neighbor.
    Diagonal cost : O(n * d) — d = degree of J coupling graph.
    """
    def __init__(self, n, J, h):
        """
        Parameters
        ----------
        n : int
            Number of spins. Matrix dimension is 2^n.
        J : (n, n) ndarray
            Symmetric coupling matrix. J[i,j] = J[j,i], J[i,i] = 0.
        h : (n,) ndarray
            Transverse field strengths. Must be > 0 for walk to be
            well-defined (zero h_i means no edge along that direction).
        """
        self.n = n
        self.J = J
        self.h = h

    def get_neighbors_and_weights(self, x):
        neighbors = [x ^ (1 << i) for i in range(self.n)]
        weights   = np.array([-self.h[i] for i in range(self.n)])
        return neighbors, weights

    def diagonal_entry(self, x):
        energy = 0.0
        for i in range(self.n):
            si = 1 - 2 * ((x >> i) & 1)
            for j in np.nonzero(self.J[i])[0]:
                if j > i:
                    sj = 1 - 2 * ((x >> int(j)) & 1)
                    energy -= self.J[i, j] * si * sj
        return energy


# ═══════════════════════════════════════════════════════════════════
# SYSTEM 2: p-body TFIM
# ═══════════════════════════════════════════════════════════════════

class PBodyTFIM(SpinSystem):
    """
    p-body Transverse Field Ising Model.

        H = -sum_{|S|=p} J_S (prod_{i in S} Z_i) - sum_i h_i X_i

    Off-diagonal part M: IDENTICAL to TFIM — single spin flips, weight -h_i.
    The p-body structure only affects the diagonal D.

    Neighbor cost : O(n)         — same as TFIM, walk is UNCHANGED.
    Diagonal cost : O(C(n, p))   — sum over all p-tuples in J_tensor.

    Parameters
    ----------
    J_tensor : dict {frozenset of p indices -> coupling value}
        Use PBodyTFIM.random_J_tensor(n, p) to generate randomly.
    """
    def __init__(self, n, J_tensor, h, p):
        self.n        = n
        self.J_tensor = J_tensor
        self.h        = h
        self.p        = p

    def get_neighbors_and_weights(self, x):
        """Identical to TFIM — p-body only affects diagonal."""
        neighbors = [x ^ (1 << i) for i in range(self.n)]
        weights   = np.array([-self.h[i] for i in range(self.n)])
        return neighbors, weights

    def diagonal_entry(self, x):
        energy = 0.0
        for indices, J_val in self.J_tensor.items():
            spin_product = 1.0
            for i in indices:
                spin_product *= (1 - 2 * ((x >> i) & 1))
            energy -= J_val * spin_product
        return energy

    @staticmethod
    def random_J_tensor(n, p, scale=1.0, seed=None):
        """
        Generate a random J_tensor for p-body interactions.

        Parameters
        ----------
        n     : number of spins
        p     : interaction order (p=2 recovers TFIM)
        scale : coupling strength
        seed  : random seed

        Returns
        -------
        dict {frozenset -> float}
        """
        rng      = np.random.default_rng(seed)
        J_tensor = {}
        for indices in combinations(range(n), p):
            J_tensor[frozenset(indices)] = rng.standard_normal() * scale
        return J_tensor


# ═══════════════════════════════════════════════════════════════════
# SYSTEM 3: Random Ising (Sherrington-Kirkpatrick)
# ═══════════════════════════════════════════════════════════════════

class RandomIsing(SpinSystem):
    """
    Sherrington-Kirkpatrick (SK) spin glass with transverse field.

        H = -sum_{i<j} J_ij Z_i Z_j - sum_i h_i X_i
        J_ij ~ N(0, J_scale^2 / n)

    Structurally identical to TFIM. Provided as a separate class
    for convenience and to enforce SK scaling of couplings.

    Neighbor cost : O(n).
    Diagonal cost : O(n * n) = O(n^2) since J is dense.
    """
    def __init__(self, n, h, J_scale=1.0, seed=None):
        """
        Parameters
        ----------
        n       : number of spins
        h       : (n,) transverse field
        J_scale : coupling scale. SK convention: J_ij ~ N(0, J_scale^2/n).
        seed    : random seed for reproducibility of J.
        """
        self.n = n
        self.h = h
        rng    = np.random.default_rng(seed)
        raw    = rng.standard_normal((n, n)) * J_scale / np.sqrt(n)
        self.J = (raw + raw.T) / 2
        np.fill_diagonal(self.J, 0)

    def get_neighbors_and_weights(self, x):
        neighbors = [x ^ (1 << i) for i in range(self.n)]
        weights   = np.array([-self.h[i] for i in range(self.n)])
        return neighbors, weights

    def diagonal_entry(self, x):
        energy = 0.0
        for i in range(self.n):
            si = 1 - 2 * ((x >> i) & 1)
            for j in range(i + 1, self.n):
                if self.J[i, j] != 0:
                    sj = 1 - 2 * ((x >> j) & 1)
                    energy -= self.J[i, j] * si * sj
        return energy


# ═══════════════════════════════════════════════════════════════════
# SYSTEM 4: XXZ model
# ═══════════════════════════════════════════════════════════════════

class XXZ(SpinSystem):
    """
    XXZ model (anisotropic Heisenberg).

        H = sum_{i<j} J_ij (X_i X_j + Y_i Y_j) + Delta * sum_{i<j} J_ij Z_i Z_j

    The XX+YY term = 2(S+_i S-_j + S-_i S+_j) flips TWO spins simultaneously
    (one up->down, one down->up), preserving total magnetization.

    Off-diagonal part M: two-spin swap moves, weight 2*J_ij.
                         Each config has O(n^2) neighbors.
    Diagonal part D:     ZZ anisotropy, weight Delta * J_ij.

    Neighbor cost : O(n^2) — all (i up, j down) pairs.
    Diagonal cost : O(n^2) — all pairs for ZZ term.

    NOTE: The walk graph is NOT the hypercube but the union of
    magnetization-preserving swap moves. Configs with all spins up or
    all down are isolated (no neighbors) and contribute only via the
    diagonal to the Hutchinson estimator.

    Parameters
    ----------
    Delta : float
        Anisotropy. Delta=1: Heisenberg. Delta=0: pure XX model.
                    Delta->inf: classical Ising limit.
    """
    def __init__(self, n, J, Delta=1.0):
        self.n     = n
        self.J     = J
        self.Delta = Delta

    def get_neighbors_and_weights(self, x):
        neighbors, weights = [], []
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                si = (x >> i) & 1
                sj = (x >> j) & 1
                # S+_i S-_j acts on |..down_i..up_j..> -> |..up_i..down_j..>
                if si == 0 and sj == 1:
                    x_new = x ^ (1 << i) ^ (1 << j)
                    neighbors.append(x_new)
                    weights.append(2.0 * self.J[i, j])
        return neighbors, np.array(weights) if weights else np.array([])

    def diagonal_entry(self, x):
        energy = 0.0
        for i in range(self.n):
            si = 1 - 2 * ((x >> i) & 1)
            for j in range(i + 1, self.n):
                if self.J[i, j] != 0:
                    sj = 1 - 2 * ((x >> j) & 1)
                    energy += self.Delta * self.J[i, j] * si * sj
        return energy


# ═══════════════════════════════════════════════════════════════════
# Modulation functions
# ═══════════════════════════════════════════════════════════════════

def f_diffusion(beta):
    """
    Modulation function for f(H) = e^{-beta H}.
    Taylor coefficient at order k: (-beta)^k / (2^k * k!)

    The 2^k factor is the GRF normalization convention from
    Choromanski et al. — it accounts for the symmetric random walk
    on the hypercube where each step divides by 2.
    """
    return lambda k: ((-beta) ** k) / (2**k * scipy.special.factorial(k))

def f_resolvent(beta):
    """
    Modulation function for f(H) = (I - beta H)^{-1}.
    Taylor coefficient at order k: beta^k (geometric series).
    Only valid when beta * ||H||_2 < 1.
    """
    return lambda k: float(beta ** k)

def f_p_step_rw(p):
    """
    Modulation function for f(H) = (I + H)^p.
    Taylor coefficient at order k: C(p, k) (binomial).
    """
    return lambda k: float(scipy.special.comb(p, k))


# ═══════════════════════════════════════════════════════════════════
# Main verification
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    beta     = 0.3
    n        = 3
    J_val    = 0.4
    h_val    = 0.3
    p_halt   = 0.3
    nb_walks = 64
    nb_hutch = 300

    # 1D chain coupling
    J = np.zeros((n, n))
    for i in range(n - 1):
        J[i, i+1] = J[i+1, i] = J_val
    h = np.full(n, h_val)

    systems = [
        TFIM(n, J, h),
        PBodyTFIM(n, PBodyTFIM.random_J_tensor(n, 3, scale=0.3, seed=1), h, p=3),
        RandomIsing(n, h, J_scale=0.5, seed=0),
        XXZ(n, J, Delta=1.0),
    ]

    for system in systems:
        system.verify(
            beta          = beta,
            p_halt        = p_halt,
            nb_walks      = nb_walks,
            nb_hutchinson = nb_hutch,
        )
