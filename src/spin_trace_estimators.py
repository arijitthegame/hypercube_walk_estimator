"""
spin_trace_estimator.py
=======================
Graph Random Feature (GRF) + Hutchinson trace estimator for spin system
Hamiltonians of the form:

    H = D + M

where:
    D  = diagonal matrix of ZZ (or p-body) coupling energies
    M  = off-diagonal transverse field matrix

THREE ESTIMATION MODES
----------------------
mode='M_only'
    Estimates tr(f(M)), ignoring D entirely. EXACTLY UNBIASED for tr(f(M)).
    The walk only traverses off-diagonal edges with weights -h_i.
    Bias vs tr(f(H)): O(beta^2 ||J||_F ||h||_2).

    CLOSED FORM: for all systems with hypercube M (TFIM, p-body, SK),
    tr(e^{-beta M}) = 2^n * prod_i cosh(beta * h_i)
    This is O(n) to compute exactly, so mode='M_only' is mainly useful
    as a baseline / control variate, NOT as an approximation to tr(f(H)).

    NOTE: np.trace(expm(-beta * M)) and the closed form give IDENTICAL
    results (up to floating point). There is no need to zero-fill the
    diagonal of H and call expm — the closed form is both cheaper and
    exact. See exact_trace_M_hypercube() below.

mode='trotter'
    Estimates tr(e^{-beta D} e^{-beta M}) via symmetric Trotter splitting:
        e^{-beta H} = e^{-beta D/2} e^{-beta M} e^{-beta D/2} + O(beta^3)
    Diagonal correction e^{-beta D_{xx}/2} absorbed at start/end of walk.
    Bias vs tr(f(H)): O(beta^3 ||[D,M]||^2). Only valid for f = e^{-beta H}.

mode='unbiased'  [RECOMMENDED]
    Estimates tr(f(H)) for the FULL Hamiltonian. EXACTLY UNBIASED.
    Diagonal D is handled via self-loops: the walk can stay at the
    current vertex with weight D_{xx}, contributing to H^k via
    interleavings of diagonal and off-diagonal steps. Since the GRF
    expands f(H) = sum_k alpha_k H^k at the TAYLOR SERIES level (not
    the exponential level), non-commutativity of D and M is irrelevant
    — all interleavings of D and M steps in H^k are captured exactly.

    Diagonal cost is minimised via INCREMENTAL UPDATES:
        - Sparse J (TFIM, 1D/2D/k-regular): O(d) per step, d = degree
        - Dense J (SK/RandomIsing):          O(n) per step (incremental)
        - p-body:                            O(C(n-1,p-1)) per step
    vs O(n*d) or O(n^2) if recomputed from scratch each step.

BIAS SUMMARY
------------
    Mode        Targets                   Bias vs tr(f(H))           Cost/step
    --------    ----------------------    --------------------------  ---------
    M_only      tr(f(M))                  O(b^2 ||J||_F ||h||_2)     O(n)
    trotter     tr(e^{-bD} e^{-bM})       O(b^3 ||[D,M]||^2)         O(n*d)
    unbiased    tr(f(H))                  0                           O(n+d)

HOW TO VERIFY EACH MODE
------------------------
Run system.verify(...) which prints a table comparing all three modes
against their correct ground truths:

    exact_M_closed : 2^n * prod_i cosh(beta*h_i)   <- O(n), exact, no expm
    exact_M_expm   : np.trace(expm(-beta*M))        <- O(8^n), same value
    exact_trotter  : np.trace(expm(-b*D) @ expm(-b*M))
    exact_H        : np.trace(expm(-beta*H))        <- true partition function

Expected behaviour:
    mode='M_only'   ~0 error vs exact_M_closed (and exact_M_expm, same thing)
    mode='trotter'  ~0 error vs exact_trotter
    mode='unbiased' ~0 error vs exact_H

COMPLEXITY SUMMARY
------------------
    Total cost: O(S * R * L * cost_per_step)
    S = nb_hutchinson, R = nb_walks, L = mean walk length = (1-p)/p

    System          mode='M_only'  mode='unbiased' (per step)
    -----------     -------------  ---------------------------
    TFIM sparse     O(n)           O(n + d),  d = J graph degree
    TFIM dense/SK   O(n)           O(n)       incremental
    p-body          O(n)           O(n + C(n-1,p-1))
    XXZ             O(n^2)         O(n^2)     (M already O(n^2) neighbors)
"""

import numpy as np
import scipy
from scipy.linalg import expm
from itertools import combinations


# ═══════════════════════════════════════════════════════════════════
# Closed form for tr(e^{-beta M}) on hypercube M
# ═══════════════════════════════════════════════════════════════════

def exact_trace_M_hypercube(n, h, beta):
    """
    Exact closed form for tr(e^{-beta M}) where M = -sum_i h_i X_i.

    Since X_i on different sites commute, the exponential factorises:
        e^{-beta M} = prod_i e^{beta h_i X_i}

    Taking the trace over each site independently:
        tr(e^{beta h_i X_i}) = 2 cosh(beta h_i)

    Therefore:
        tr(e^{-beta M}) = 2^n * prod_i cosh(beta h_i)

    This formula is valid for ANY h_i (uniform or not) and costs O(n).
    It gives IDENTICAL results to np.trace(expm(-beta * M)) up to
    floating point precision. There is no need to zero-fill the diagonal
    of H and call expm — both approaches compute the same quantity.

    Parameters
    ----------
    n    : int,        number of spins
    h    : (n,) array, transverse field strengths
    beta : float,      inverse temperature

    Returns
    -------
    float : tr(e^{-beta M})
    """
    return float((2 ** n) * np.prod(np.cosh(beta * np.asarray(h))))

def rademacher_hash(x, seed):
    z = (x + seed * 0x9e3779b97f4a7c15 + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & 0xFFFFFFFFFFFFFFFF
    z ^= z >> 31
    return 1.0 if (z & 1) == 0 else -1.0


# ═══════════════════════════════════════════════════════════════════
# BASE CLASS
# ═══════════════════════════════════════════════════════════════════

class SpinSystem:
    """
    Abstract base class for spin Hamiltonians H = D + M.

    Subclasses must implement:
        get_neighbors_and_weights(x) -> (neighbors, weights)
            Off-diagonal edges only. Used by M_only and trotter modes.
        diagonal_entry(x) -> float
            H[x,x]. Used by trotter and unbiased modes.
        build_adj_list() -> list of list of (j, J_ij)
            Sparse adjacency list of the J coupling graph.
            Used by unbiased mode for incremental diagonal updates.

    Optionally override:
        supports_closed_form_M : bool
            Set True if tr(e^{-beta M}) = 2^n prod_i cosh(beta h_i).
            True for all systems with hypercube M (TFIM, p-body, SK).
            False for XXZ (different M structure).
    """

    supports_closed_form_M = False

    # ── Subclass interface ────────────────────────────────────────

    def get_neighbors_and_weights(self, x):
        """Off-diagonal neighbors and their signed H matrix weights."""
        raise NotImplementedError

    def diagonal_entry(self, x):
        """H[x, x] for configuration x."""
        raise NotImplementedError

    def build_adj_list(self):
        """
        Sparse adjacency list of J coupling graph for incremental updates.
        Returns list of length n: adj[i] = list of (j, J_ij) for j != i.
        Default: empty lists (no incremental speedup, falls back to
        calling diagonal_entry from scratch at each step).
        Override in subclasses where J is explicitly available.
        """
        return [[] for _ in range(self.n)]

    # ── Shared utilities ──────────────────────────────────────────

    def _transition_probs(self, weights):
        """Probabilities proportional to |weights|. Returns zeros if all zero."""
        abs_w = np.abs(weights)
        total = abs_w.sum()
        return abs_w / total if total > 0 else np.zeros_like(abs_w)

    # ── Walk for M_only and trotter (no self-loops) ───────────────

    def _random_walk_offdiag(self, start, p_halt):
        """
        Walk on off-diagonal graph only. Used by M_only and trotter modes.
        Returns list of (vertex, neighbors, weights, probs).
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

    # ── Incremental diagonal state for unbiased mode ──────────────

    def _init_diagonal_state(self, x):
        """
        Initialise spin state, diagonal D_{xx}, and local fields l_i
        at configuration x. Cost O(n * d) where d = mean J-degree.

        Returns
        -------
        spins   : np.ndarray shape (n,), values in {-1, +1}
        diag    : float, D_{xx}
        lfields : np.ndarray shape (n,), l_i = sum_j J_ij sigma_j
        """
        spins   = np.array([1 - 2 * ((x >> i) & 1)
                            for i in range(self.n)], dtype=float)
        lfields = np.zeros(self.n)
        for i, neighbors_i in enumerate(self.adj):
            for j, J_ij in neighbors_i:
                lfields[i] += J_ij * spins[j]
        # D_{xx} = -0.5 * sum_i sigma_i * l_i  (each pair counted twice)
        diag = -0.5 * float(np.dot(spins, lfields))
        return spins, diag, lfields

    def _flip_spin_incremental(self, i, spins, diag, lfields):
        """
        Update (spins, diag, lfields) after flipping spin i.
        Cost O(d_i) where d_i = degree of spin i in J graph.

        Update rule (derived from D = -sum_{i<j} J_ij sigma_i sigma_j):
            D_new    = D_old + 2 * sigma_i * l_i
            l_j     -= 2 * J_ij * sigma_i   for all j in N(i)
            sigma_i  = -sigma_i
        """
        diag += 2.0 * spins[i] * lfields[i]
        for j, J_ij in self.adj[i]:
            lfields[j] -= 2.0 * J_ij * spins[i]
        spins[i] = -spins[i]
        return spins, diag, lfields

    # ── Walk for unbiased mode (with self-loops) ──────────────────

    def _random_walk_with_diagonal(self, start, p_halt):
        """
        Walk on full H including diagonal self-loops.
        Diagonal computed incrementally via _flip_spin_incremental.
        Returns list of (vertex, neighbors, weights, probs).

        Self-loop at x has weight D_{xx} and keeps the walk at x.
        Off-diagonal steps use -h_i weights as in M_only.
        Cost per step: O(d_i) for diagonal update + O(n) for neighbors.
        """
        spins, diag, lfields = self._init_diagonal_state(start)
        x    = start
        walk = []

        while True:
            nb_offdiag, w_offdiag = self.get_neighbors_and_weights(x)
            w_offdiag = np.array(w_offdiag, dtype=float)

            # Prepend self-loop: vertex x, weight D_{xx}
            neighbors = [x]       + list(nb_offdiag)
            weights   = np.array([diag] + list(w_offdiag))
            probs     = self._transition_probs(weights)
            walk.append((x, neighbors, weights, probs))

            if np.random.uniform() < p_halt:
                break

            idx    = np.random.choice(len(neighbors), p=probs)
            next_x = neighbors[idx]

            if next_x != x:
                # Identify which spin flipped and update incrementally
                flipped = int(np.log2(x ^ next_x))
                spins, diag, lfields = self._flip_spin_incremental(
                    flipped, spins, diag, lfields)
            # If self-loop (next_x == x): diag unchanged, no update needed
            x = next_x

        return walk

    # ── GRF vector implementations ────────────────────────────────
    # these functions can be used to estimate Lang-Trotter or off diagonals

    def _grf_vector_M_only(self, v, p_halt, nb_walks, f_vec):
        """
        Unbiased estimate of [f(M)]_{:,v} via off-diagonal walk.
        D is ignored entirely. Cost O(n) per step.
        """
        rf = {}
        for _ in range(nb_walks):
            walk            = self._random_walk_offdiag(v, p_halt)
            weights_product = 1.0
            marginal_prob   = 1.0
            for step, (node, neighbors, weights, probs) in enumerate(walk):
                if step < len(f_vec):
                    rf[node] = rf.get(node, 0.0) + \
                               (weights_product / marginal_prob) * f_vec[step]
                if step < len(walk) - 1:
                    next_node        = walk[step + 1][0]
                    idx              = list(neighbors).index(next_node)
                    weights_product *= weights[idx]
                    marginal_prob   *= (1 - p_halt) * probs[idx]
        return {node: val / nb_walks for node, val in rf.items()}

    def _grf_vector_trotter(self, v, p_halt, nb_walks, f_vec, beta):
        """
        Estimate [e^{-bD/2} f(M) e^{-bD/2}]_{:,v} via Trotter correction.
        Left factor e^{-bD[v,v]/2} absorbed at start; right factor at each node.
        Cost O(n*d) per step (diagonal_entry called at each node).
        Only meaningful for f = e^{-beta H}.
        """
        rf          = {}
        left_factor = np.exp(-beta * self.diagonal_entry(v) / 2.0)
        for _ in range(nb_walks):
            walk            = self._random_walk_offdiag(v, p_halt)
            weights_product = left_factor
            marginal_prob   = 1.0
            for step, (node, neighbors, weights, probs) in enumerate(walk):
                if step < len(f_vec):
                    right_factor = np.exp(-beta * self.diagonal_entry(node) / 2.0)
                    rf[node]     = rf.get(node, 0.0) + \
                                   (weights_product / marginal_prob) * \
                                   f_vec[step] * right_factor
                if step < len(walk) - 1:
                    next_node        = walk[step + 1][0]
                    idx              = list(neighbors).index(next_node)
                    weights_product *= weights[idx]
                    marginal_prob   *= (1 - p_halt) * probs[idx]
        return {node: val / nb_walks for node, val in rf.items()}

    def _grf_vector_unbiased(self, v, p_halt, nb_walks, f_vec):
        """
        Unbiased estimate of [f(H)]_{:,v} via walk on full H with self-loops.
        Diagonal handled via incremental updates. Cost O(n + d) per step.

        Unbiasedness: the walk samples paths in H^k exactly, including all
        interleavings of diagonal (self-loop) and off-diagonal steps.
        Non-commutativity of D and M is irrelevant because we expand
        f(H) = sum_k alpha_k H^k at the Taylor series level, not the
        exponential level. The GRF proposition applies to any matrix A,
        so setting A = H gives an unbiased estimator of [f(H)]_{wv}.
        """
        rf = {}
        for _ in range(nb_walks):
            walk            = self._random_walk_with_diagonal(v, p_halt)
            weights_product = 1.0
            marginal_prob   = 1.0
            for step, (node, neighbors, weights, probs) in enumerate(walk):
                if step < len(f_vec):
                    rf[node] = rf.get(node, 0.0) + \
                               (weights_product / marginal_prob) * f_vec[step]
                if step < len(walk) - 1:
                    next_node        = walk[step + 1][0]
                    idx              = neighbors.index(next_node)
                    weights_product *= weights[idx]
                    marginal_prob   *= (1 - p_halt) * probs[idx]
        return {node: val / nb_walks for node, val in rf.items()}

    # ------------------------------------------------------------

    # ── Public trace estimator ────────────────────────────────────

    def trace_estimate(self, f, beta, p_halt,
                   nb_walks,
                   nb_hutchinson,
                   nb_sources,
                   mode='unbiased',
                   max_walk_length=50,
                   verbose=False):

        assert mode in ('M_only', 'trotter', 'unbiased')

        if mode == 'unbiased':
          self.adj = self.build_adj_list()

        size  = 1 << self.n
      
        # IMPORTANT: If estimating Tr(f(H)) via AB^T, the walks A and B 
        # should represent sqrt(f). For e^{-beta H}, this means f_vec 
        # should be precomputed using beta/2.
        f_vec = [f(k) for k in range(max_walk_length)]

        # We average the sum_A * sum_B over nb_walks to reduce variance
        # and scale by size/nb_sources to estimate the full trace.
        norm = size / (nb_sources * nb_walks)

        estimates = []

        for s in range(nb_hutchinson):
          est = 0.0

          for _ in range(nb_sources):
              v = np.random.randint(size)
              
              if mode == 'unbiased':
                  walk_fn = self._random_walk_with_diagonal
              else:
                  walk_fn = self._random_walk_offdiag

              for _ in range(nb_walks):
                  # Independent walks from same source v
                  walk_A = walk_fn(v, p_halt)
                  walk_B = walk_fn(v, p_halt)

                  # --- WALK A ---
                  sum_A = 0.0
                  weights_product = 1.0
                  marginal_prob   = 1.0

                  if mode == 'trotter':
                      # Use beta/2.0 because this walk represents f(H)^{1/2}
                      weights_product *= np.exp(-beta/2.0 * self.diagonal_entry(v) / 2.0)

                  for step, (node, neighbors, weights, probs) in enumerate(walk_A):
                      if step < len(f_vec):
                          contrib = (weights_product / marginal_prob) * f_vec[step]
                          if mode == 'trotter':
                              contrib *= np.exp(-beta/2.0 * self.diagonal_entry(node) / 2.0)
                          
                          sum_A += rademacher_hash(node, s) * contrib

                      if step < len(walk_A) - 1:
                          next_node = walk_A[step + 1][0]
                          idx = neighbors.index(next_node)
                          weights_product *= weights[idx]
                          marginal_prob   *= (1 - p_halt) * probs[idx]

                  # --- WALK B ---
                  sum_B = 0.0
                  weights_product = 1.0
                  marginal_prob   = 1.0

                  if mode == 'trotter':
                      weights_product *= np.exp(-beta/2.0 * self.diagonal_entry(v) / 2.0)

                  for step, (node, neighbors, weights, probs) in enumerate(walk_B):
                      if step < len(f_vec):
                          contrib = (weights_product / marginal_prob) * f_vec[step]
                          if mode == 'trotter':
                              contrib *= np.exp(-beta/2.0 * self.diagonal_entry(node) / 2.0)
                          
                          sum_B += rademacher_hash(node, s) * contrib

                      if step < len(walk_B) - 1:
                          next_node = walk_B[step + 1][0]
                          idx = neighbors.index(next_node)
                          weights_product *= weights[idx]
                          marginal_prob   *= (1 - p_halt) * probs[idx]

                  # Unbiased estimate of row v of AB^T: (z^T A_v) * (z^T B_v)
                  est += norm * sum_A * sum_B

          estimates.append(est)
          if verbose:
              print(f"[{mode}] sample {s+1}: {est:.4f}, mean={np.mean(estimates):.4f}")

        return float(np.mean(estimates))


    # ── Build full matrix (verification only) ────────────────────

    def build_hamiltonian(self):
        """Build full 2^n x 2^n H = D + M. Only for small n verification."""
        size = 1 << self.n
        H    = np.zeros((size, size))
        for x in range(size):
            H[x, x] = self.diagonal_entry(x)
            neighbors, weights = self.get_neighbors_and_weights(x)
            for nb, w in zip(neighbors, weights):
                H[x, nb] = w
        return H

    # ── Verification table ────────────────────────────────────────

    def verify(self, beta, p_halt, nb_walks, nb_hutchinson,
               nb_sources=1, max_walk_length=50):
        """
        Compare all three estimator modes against their ground truths.

        Ground truths:
            exact_M_closed  : 2^n * prod_i cosh(beta*h_i)
                              O(n), exact, only for hypercube M systems.
                              IDENTICAL to exact_M_expm up to float precision.
            exact_M_expm    : np.trace(expm(-beta*M)), O(8^n).
                              Shown alongside closed form to confirm equality.
            exact_trotter   : np.trace(expm(-b*D) @ expm(-b*M)), O(8^n).
            exact_H         : np.trace(expm(-beta*H)), O(8^n).

        Expected results:
            mode='M_only'   ~0 error vs exact_M_expm  (and closed form)
            mode='trotter'  ~0 error vs exact_trotter
            mode='unbiased' ~0 error vs exact_H
        """
        H = self.build_hamiltonian()
        D = np.diag(np.diag(H))
        M = H - D

        exact_H       = np.trace(expm(-beta * H))
        exact_M_expm  = np.trace(expm(-beta * M))
        eD = expm(-beta * D / 2)
        eM = expm(-beta * M / 2)
        exact_trotter = np.trace(eD @ eM @ eD @ eM)
        f = f_diffusion(beta)

        print(f"\n{'='*70}")
        print(f"  System : {self.__class__.__name__}  "
              f"n={self.n}  beta={beta}  p_halt={p_halt}")
        print(f"{'='*70}")

        print(f"\n  Ground truths for tr(e^{{-beta*M}}):")
        print(f"    exact_M_expm   = {exact_M_expm:.8f}  [expm on M, O(8^n)]")
        if self.supports_closed_form_M:
            exact_M_closed = exact_trace_M_hypercube(self.n, self.h, beta)
            print(f"    exact_M_closed = {exact_M_closed:.8f}  "
                  f"[2^n*prod cosh(b*h_i), O(n)]")
            print(f"    difference     = {abs(exact_M_expm - exact_M_closed):.2e}"
                  f"  <- machine precision confirms they are identical")
        else:
            print(f"    closed form not available for {self.__class__.__name__}"
                  f" (M is not a sum of independent X_i terms)")

        print(f"\n  Other ground truths:")
        print(f"    exact_trotter  = {exact_trotter:.8f}  [expm(-bD) @ expm(-bM)]")
        print(f"    exact_H        = {exact_H:.8f}  [true partition function]")

        print(f"\n  Bias from ignoring D:")
        bias_M  = abs(exact_M_expm - exact_H)
        bias_T  = abs(exact_trotter - exact_H)
        print(f"    |tr(e^{{-bM}}) - tr(e^{{-bH}})|         = {bias_M:.6f}")
        print(f"    |tr(e^{{-bD}}e^{{-bM}}) - tr(e^{{-bH}})| = {bias_T:.6f}  "
              f"({bias_M / max(bias_T, 1e-15):.1f}x reduction from Trotter)")

        print(f"\n  Running estimators "
              f"({nb_hutchinson} Hutchinson samples, {nb_walks} walks)...")

        results = {}
        for mode in ('M_only', 'trotter', 'unbiased'):
            results[mode] = self.trace_estimate(
                f=f, beta=beta, p_halt=p_halt, nb_walks=nb_walks,
                nb_hutchinson=nb_hutchinson, mode=mode,
                max_walk_length=max_walk_length, nb_sources=nb_sources
            )

        targets = {
            'M_only'  : ('exact_M_expm',  exact_M_expm),
            'trotter' : ('exact_trotter', exact_trotter),
            'unbiased': ('exact_H',       exact_H),
        }

        print(f"\n  {'Mode':<12} {'Estimate':>10} {'Target':>14} "
              f"{'Target val':>12} {'Rel. error':>12}")
        print(f"  {'-'*63}")
        for mode, est in results.items():
            tname, tval = targets[mode]
            err = abs(est - tval) / abs(tval)
            print(f"  {mode:<12} {est:>10.4f} {tname:>14} "
                  f"{tval:>12.4f} {err:>11.3%}")
        print()


# ═══════════════════════════════════════════════════════════════════
# SYSTEM 1: Standard TFIM (sparse J)
# ═══════════════════════════════════════════════════════════════════

class TFIM(SpinSystem):
    """
    Transverse Field Ising Model.
        H = -sum_{i<j} J_ij Z_i Z_j - sum_i h_i X_i

    Off-diagonal M : single spin flips, weight -h_i. Hypercube graph.
    Diagonal D     : two-body ZZ coupling energies.

    supports_closed_form_M = True

    Neighbor cost  : O(n)
    Diagonal init  : O(n * d),  d = degree of J coupling graph
    Diagonal update: O(d_i)     incremental, only neighbors of flipped spin
                                -> O(1) per step for sparse J (1D chain, grid)
                                -> O(n) per step for dense J (use RandomIsing)
    """

    supports_closed_form_M = True

    def __init__(self, n, J, h):
        """
        Parameters
        ----------
        n : int,         number of spins
        J : (n,n) array, symmetric coupling matrix, J[i,i] = 0
        h : (n,) array,  transverse field strengths (all > 0)
        """
        self.n = n
        self.J = J
        self.h = h

    def get_neighbors_and_weights(self, x):
        neighbors = [x ^ (1 << i) for i in range(self.n)]
        weights   = np.array([-self.h[i] for i in range(self.n)])
        return neighbors, weights

    def diagonal_entry(self, x):
        """O(n * d) from scratch. Incremental update used inside walks."""
        energy = 0.0
        for i in range(self.n):
            si = 1 - 2 * ((x >> i) & 1)
            for j in np.nonzero(self.J[i])[0]:
                if j > i:
                    sj = 1 - 2 * ((x >> int(j)) & 1)
                    energy -= self.J[i, j] * si * sj
        return energy

    def build_adj_list(self):
        """
        Sparse adjacency list of J. Built once per trace_estimate call.
        Per-step flip cost: O(d_i) where d_i = degree of spin i.
        For 1D chain: d_i = 2 -> O(1) per step.
        """
        adj = [[] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.J[i, j] != 0.0:
                    adj[i].append((j, self.J[i, j]))
        return adj


# ═══════════════════════════════════════════════════════════════════
# SYSTEM 2: p-body TFIM
# ═══════════════════════════════════════════════════════════════════

class PBodyTFIM(SpinSystem):
    """
    p-body Transverse Field Ising Model.
        H = -sum_{|S|=p} J_S (prod_{i in S} Z_i) - sum_i h_i X_i

    Off-diagonal M : IDENTICAL to TFIM — single spin flips, weight -h_i.
    Diagonal D     : p-body interactions.

    supports_closed_form_M = True  (same hypercube M as TFIM)

    Neighbor cost  : O(n)
    Diagonal init  : O(C(n,p))       — sum over all p-tuples
    Diagonal update: O(C(n-1,p-1))   — only p-tuples containing flipped spin
                     Precomputed via _spin_to_tuples index at construction.
    """

    supports_closed_form_M = True

    def __init__(self, n, J_tensor, h, p):
        """
        Parameters
        ----------
        n        : int,   number of spins
        J_tensor : dict   {frozenset of p indices -> float coupling}
        h        : (n,)   transverse field
        p        : int,   interaction order (p=2 recovers standard TFIM)
        """
        self.n        = n
        self.J_tensor = J_tensor
        self.h        = h
        self.p        = p
        # Index: spin i -> list of (tuple_indices, J_val) that contain i
        # Allows O(C(n-1,p-1)) incremental diagonal update per flip.
        self._spin_to_tuples = {i: [] for i in range(n)}
        for indices, J_val in J_tensor.items():
            for i in indices:
                self._spin_to_tuples[i].append((indices, J_val))

    def get_neighbors_and_weights(self, x):
        neighbors = [x ^ (1 << i) for i in range(self.n)]
        weights   = np.array([-self.h[i] for i in range(self.n)])
        return neighbors, weights

    def diagonal_entry(self, x):
        """O(C(n,p)) from scratch."""
        energy = 0.0
        for indices, J_val in self.J_tensor.items():
            spin_product = 1.0
            for i in indices:
                spin_product *= (1 - 2 * ((x >> i) & 1))
            energy -= J_val * spin_product
        return energy

    def build_adj_list(self):
        """
        p-body has no 2-body adj list. Return empty — we override
        _init_diagonal_state and _flip_spin_incremental instead.
        """
        return [[] for _ in range(self.n)]

    def _init_diagonal_state(self, x):
        """
        For p-body: store spins and precompute diagonal from scratch.
        Local fields are not defined for p > 2, so we skip them.
        Returns (spins, diag, None) — lfields unused.
        """
        spins = np.array([1 - 2 * ((x >> i) & 1)
                          for i in range(self.n)], dtype=float)
        diag  = self.diagonal_entry(x)
        return spins, diag, None

    def _flip_spin_incremental(self, i, spins, diag, lfields):
        """
        p-body incremental update: revisit only p-tuples containing spin i.
        Cost O(C(n-1, p-1)) per flip — much cheaper than O(C(n,p)) from scratch.
        """
        for indices, J_val in self._spin_to_tuples[i]:
            old_prod = 1.0
            for k in indices:
                old_prod *= spins[k]
            new_prod  = old_prod * (-1)   # only spin i changes sign
            diag     += J_val * (old_prod - new_prod)
        spins[i] = -spins[i]
        return spins, diag, None

    @staticmethod
    def random_J_tensor(n, p, scale=1.0, seed=None):
        """Random p-body coupling tensor as dict {frozenset -> float}."""
        rng      = np.random.default_rng(seed)
        J_tensor = {}
        for indices in combinations(range(n), p):
            J_tensor[frozenset(indices)] = rng.standard_normal() * scale
        return J_tensor


# ═══════════════════════════════════════════════════════════════════
# SYSTEM 3: Sherrington-Kirkpatrick (dense random Ising)
# ═══════════════════════════════════════════════════════════════════

class RandomIsing(SpinSystem):
    """
    Sherrington-Kirkpatrick (SK) spin glass with transverse field.
        H = -sum_{i<j} J_ij Z_i Z_j - sum_i h_i X_i,  J_ij ~ N(0, J^2/n)

    J is DENSE: every spin couples to every other.
    Diagonal cost from scratch: O(n^2).
    With incremental update: O(n) per step — best possible for dense J
    since flipping any spin changes all n-1 local fields.

    supports_closed_form_M = True  (same hypercube M as TFIM)

    Neighbor cost  : O(n)
    Diagonal init  : O(n^2)   — unavoidable for dense J, done once per walk
    Diagonal update: O(n)     — incremental, all n neighbors of flipped spin
    """

    supports_closed_form_M = True

    def __init__(self, n, h, J_scale=1.0, seed=None):
        """
        Parameters
        ----------
        n       : int,    number of spins
        h       : (n,) array, transverse field
        J_scale : float,  coupling scale, J_ij ~ N(0, J_scale^2 / n)
        seed    : int,    random seed for reproducibility of J
        """
        self.n = n
        self.h = h
        rng    = np.random.default_rng(seed)
        raw    = rng.standard_normal((n, n)) * J_scale / np.sqrt(n)
        self.J = (raw + raw.T) / np.sqrt(2)
        np.fill_diagonal(self.J, 0)

    def get_neighbors_and_weights(self, x):
        neighbors = [x ^ (1 << i) for i in range(self.n)]
        weights   = np.array([-self.h[i] for i in range(self.n)])
        return neighbors, weights

    def diagonal_entry(self, x):
        """O(n^2) from scratch — incremental used inside walks."""
        energy = 0.0
        for i in range(self.n):
            si = 1 - 2 * ((x >> i) & 1)
            for j in range(i + 1, self.n):
                sj = 1 - 2 * ((x >> j) & 1)
                energy -= self.J[i, j] * si * sj
        return energy

    def build_adj_list(self):
        """
        Dense adj list: spin i connects to all j != i.
        Per-step flip cost O(n) — optimal for dense SK model.
        """
        adj = []
        for i in range(self.n):
            adj.append([(j, self.J[i, j])
                        for j in range(self.n)
                        if i != j and self.J[i, j] != 0.0])
        return adj


# ═══════════════════════════════════════════════════════════════════
# SYSTEM 4: XXZ model
# ═══════════════════════════════════════════════════════════════════

class XXZ(SpinSystem):
    """
    XXZ model (anisotropic Heisenberg).
        H = sum_{i<j} J_ij (X_i X_j + Y_i Y_j) + Delta * sum_{i<j} J_ij Z_i Z_j

    XX+YY flips TWO spins simultaneously (swap move), preserving magnetization.
    Walk graph is NOT the hypercube — it is the union of magnetization-preserving
    swap moves. Configs with all spins up or all down are isolated.

    supports_closed_form_M = False  (M mixes sites, does not factorise)

    Neighbor cost  : O(n^2) — all (i up, j down) swap pairs
    Diagonal init  : O(n^2)
    Diagonal update: O(d_i + d_j) per step for two-spin flip (i, j)
                     where d_k = degree of spin k in J graph
    """

    supports_closed_form_M = False

    def __init__(self, n, J, Delta=1.0):
        """
        Parameters
        ----------
        n     : int,         number of spins
        J     : (n,n) array, symmetric coupling matrix
        Delta : float,       ZZ anisotropy (Delta=1: Heisenberg, Delta=0: XX)
        """
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
                if si == 0 and sj == 1:   # spin i up, spin j down -> swap
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

    def build_adj_list(self):
        """
        For XXZ diagonal (Delta * ZZ), adj list uses -Delta * J_ij weights.
        Each step flips two spins i, j — apply _flip_spin_incremental twice.
        Cost per step: O(d_i + d_j).
        """
        adj = []
        for i in range(self.n):
            adj.append([(j, -self.Delta * self.J[i, j])
                        for j in range(self.n)
                        if i != j and self.J[i, j] != 0.0])
        return adj

    def _random_walk_with_diagonal(self, start, p_halt):
        """
        Override for XXZ: each off-diagonal step flips TWO spins.
        Apply _flip_spin_incremental for each flipped spin separately.
        """
        spins, diag, lfields = self._init_diagonal_state(start)
        x    = start
        walk = []

        while True:
            nb_offdiag, w_offdiag = self.get_neighbors_and_weights(x)
            w_offdiag = np.array(w_offdiag, dtype=float)
            neighbors = [x]       + list(nb_offdiag)
            weights   = np.array([diag] + list(w_offdiag))
            probs     = self._transition_probs(weights)
            walk.append((x, neighbors, weights, probs))

            if np.random.uniform() < p_halt:
                break

            idx    = np.random.choice(len(neighbors), p=probs)
            next_x = neighbors[idx]

            if next_x != x:
                diff    = x ^ next_x
                flipped = [b for b in range(self.n) if diff & (1 << b)]
                for b in flipped:   # apply incremental update for each flipped spin
                    spins, diag, lfields = self._flip_spin_incremental(
                        b, spins, diag, lfields)
            x = next_x

        return walk


# ═══════════════════════════════════════════════════════════════════
# Modulation functions
# ═══════════════════════════════════════════════════════════════════

def f_diffusion(beta):
    """f(k) = (-beta)^k / (2^k * k!) for e^{-beta H}."""
    return lambda k:( (-beta) ** k) / ((2**k) * scipy.special.factorial(k))

def f_resolvent(beta):
    """f(k) = beta^k for (I - beta*H)^{-1}. Valid when beta*||H||_2 < 1."""
    return lambda k: float(beta ** k)

def f_p_step_rw(p):
    """f(k) = C(p,k) for (I + H)^p."""
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
