import numpy as np

# baseline Stochastic Lanczos Quadrature
def lanczos(Hv, n, m, v0):
    """
    Lanczos algorithm.

    Hv: function(v) -> H @ v
    n: dimension
    m: number of Lanczos steps
    v0: initial vector (normalized)

    Returns:
    - alpha: diagonal entries
    - beta: off-diagonal entries
    - Q: Lanczos basis (optional, can omit if memory matters)
    """
    Q = np.zeros((n, m))
    alpha = np.zeros(m)
    beta = np.zeros(m - 1)

    q = v0 / np.linalg.norm(v0)
    Q[:, 0] = q

    w = Hv(q)
    alpha[0] = np.dot(q, w)
    w = w - alpha[0] * q

    for j in range(1, m):
        beta[j - 1] = np.linalg.norm(w)
        if beta[j - 1] < 1e-12:
            # early termination
            return alpha[:j], beta[:j-1], Q[:, :j]

        q_next = w / beta[j - 1]
        Q[:, j] = q_next

        w = Hv(q_next) - beta[j - 1] * q
        alpha[j] = np.dot(q_next, w)
        w = w - alpha[j] * q_next

        q = q_next

    return alpha, beta, Q


def slq_trace_exp(Hv, n, beta_param, m=20, num_samples=20):
    """
    Estimate Tr(exp(-beta * H)) using Stochastic Lanczos Quadrature.

    Hv: function(v) -> H @ v
    n: dimension
    beta_param: inverse temperature
    m: Lanczos steps
    num_samples: number of probe vectors
    """
    trace_est = 0.0

    for _ in range(num_samples):
        # Rademacher vector (+1 / -1)
        z = np.random.choice([-1.0, 1.0], size=n)
        z = z / np.linalg.norm(z)

        alpha, beta, _ = lanczos(Hv, n, m, z)

        k = len(alpha)

        # Build tridiagonal matrix T
        T = np.diag(alpha)
        if k > 1:
            T += np.diag(beta[:k-1], 1)
            T += np.diag(beta[:k-1], -1)

        # Diagonalize small T
        eigvals, eigvecs = np.linalg.eigh(T)

        # e1^T f(T) e1 = sum w_i^2 f(lambda_i)
        e1 = eigvecs[0, :]
        weights = e1**2

        f_vals = np.exp(-beta_param * eigvals)
        estimate = np.sum(weights * f_vals)

        # scale by dimension
        trace_est += estimate

    return n * trace_est / num_samples
