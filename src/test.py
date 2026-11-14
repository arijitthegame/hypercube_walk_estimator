import matplotlib.pyplot as plt
import time

from spin_trace_estimators import *

def run_convergence_tests(system_class, n_spins, beta=0.2, p_halt=0.3):
    """
    Runs tests on a spin system across varying numbers of walks and Hutchinson samples
    to demonstrate empirically that the trace estimation error decreases with more sampling.
    """
    print(f"==================================================")
    print(f"Running Convergence Analysis: {system_class.__name__} (N={n_spins})")
    print(f"==================================================\n")

    # Initialize system with simple 1D coupling
    h = np.full(n_spins, 0.3)
    if system_class == PBodyTFIM:
        J_tensor = PBodyTFIM.random_J_tensor(n_spins, 3, scale=0.3, seed=42)
        sys = PBodyTFIM(n_spins, J_tensor, h, p=3)
    elif system_class == RandomIsing:
        sys = RandomIsing(n_spins, h, J_scale=0.4, seed=42)
    else:
        J = np.zeros((n_spins, n_spins))
        for i in range(n_spins - 1):
            J[i, i+1] = J[i+1, i] = 0.4
        if system_class == XXZ:
            sys = XXZ(n_spins, J, Delta=1.0)
        else:
            sys = TFIM(n_spins, J, h)

    # Exact Partition Function Ground Truth
    H = sys.build_hamiltonian()
    exact_val = np.trace(expm(-beta * H))
    f = f_diffusion(beta)

    # 1. Test Convergence vs Number of Walks (R)
    walk_sizes = [8, 16, 32, 64, 128]
    fixed_s = 100
    errors_vs_r = []

    print(f"Test 1: Error vs Walk Count (R) [Holding Hutchinson S={fixed_s}]")
    for r in walk_sizes:
        est = sys.trace_estimate(f=f, beta=beta, p_halt=p_halt, nb_walks=r, nb_hutchinson=fixed_s, nb_sources=1, mode='unbiased')
        rel_err = abs(est - exact_val) / exact_val
        errors_vs_r.append(rel_err)
        print(f"  Walks (R) = {r:3d} | Estimate = {est:10.5f} | Exact = {exact_val:10.5f} | Rel. Error = {rel_err:7.2%}")

    # 2. Test Convergence vs Hutchinson Samples (S)
    hutch_sizes = [10, 50, 100, 200, 400]
    fixed_r = 32
    errors_vs_s = []

    print(f"\nTest 2: Error vs Hutchinson Samples (S) [Holding Walks R={fixed_r}]")
    for s in hutch_sizes:
        est = sys.trace_estimate(f=f, beta=beta, p_halt=p_halt, nb_walks=fixed_r, nb_hutchinson=s, nb_sources=1, mode='unbiased')
        rel_err = abs(est - exact_val) / exact_val
        errors_vs_s.append(rel_err)
        print(f"  Hutch (S) = {s:3d} | Estimate = {est:10.5f} | Exact = {exact_val:10.5f} | Rel. Error = {rel_err:7.2%}")

    # --- Plotting Results ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(walk_sizes, errors_vs_r, 'o-', color='tab:blue', linewidth=2)
    ax1.set_title("Error vs. Number of Walks (R)")
    ax1.set_xlabel("Number of Walks (R)")
    ax1.set_ylabel("Relative Estimation Error")
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    ax2.plot(hutch_sizes, errors_vs_s, 's-', color='tab:orange', linewidth=2)
    ax2.set_title("Error vs. Hutchinson Samples (S)")
    ax2.set_xlabel("Hutchinson Samples (S)")
    ax2.set_ylabel("Relative Estimation Error")
    ax2.set_yscale('log')
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.suptitle(f"Convergence Analysis for {system_class.__name__} (N={n_spins})", fontsize=14)
    plt.tight_layout()
    plt.show()
