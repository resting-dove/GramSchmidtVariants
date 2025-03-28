import functools
import json

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import time
import numpy
import matplotlib.pyplot as plt
from generate_matrix import random_sparse, grcar


def trial_results(ns: list):
    results = {n: {
        "ts": [],
        "loss of orthogonality": [],
    } for n in ns}
    return results


if __name__ == "__main__":
    n = 5000
    rng = jax.random.PRNGKey(77)
    numpy_rng = numpy.random.default_rng(777)
    matrix = "grcar"

    if matrix == "random_sparse":
        A_scipy = random_sparse(numpy_rng, n, density=0.1)
    elif matrix == "grcar":
        A_scipy = grcar(n)
    else:
        raise RuntimeError("Matrix not found")
    b = jax.random.normal(rng, (n,), dtype=jnp.float64)
    b_scipy = numpy.array(b)
    b_length = numpy.linalg.norm(b)

    num_matvecss = [50, 200, 400, 900]
    reps = 3
    from krylov_basis import arnoldi

    ortho_mgs = trial_results(num_matvecss)
    ortho_trunc = trial_results(num_matvecss)
    ortho_cgs = trial_results(num_matvecss)
    ortho_cgs_projreortho = trial_results(num_matvecss)
    ortho_cgs_reortho = trial_results(num_matvecss)

    for num_matvecs in num_matvecss:
        for i in range(reps):
            start = time.time()
            w, Q, H, breakdown = arnoldi(A_scipy, b_scipy / b_length, num_matvecs, reortho=False, mgs=True, proj=False,
                                         trunc=np.inf)
            end = time.time()
            ortho = np.linalg.norm(np.eye(num_matvecs, num_matvecs) - Q[:, :num_matvecs + 1].T @ Q[:, :num_matvecs + 1])
            ortho_mgs[num_matvecs]["ts"].append(end - start)
            ortho_mgs[num_matvecs]["loss of orthogonality"].append(ortho)
            print(f"MGS time taken: {end - start:.6f} seconds, n={num_matvecs}, orthogonality: {ortho}")

            start = time.time()
            w, Q, H, breakdown = arnoldi(A_scipy, b_scipy / b_length, num_matvecs, reortho=False, mgs=True, proj=False, trunc=2)
            end = time.time()
            ortho = np.linalg.norm(np.eye(num_matvecs, num_matvecs) - Q[:, :num_matvecs + 1].T @ Q[:, :num_matvecs + 1])
            ortho_trunc[num_matvecs]["ts"].append(end - start)
            ortho_trunc[num_matvecs]["loss of orthogonality"].append(ortho)
            print(f"Trunc time taken: {end - start:.6f} seconds, n={num_matvecs}, orthogonality: {ortho}")

            start = time.time()
            w, Q, H, breakdown = arnoldi(A_scipy, b_scipy / b_length, num_matvecs, reortho=False, mgs=False, proj=False,
                                         trunc=np.inf)
            end = time.time()
            ortho = np.linalg.norm(np.eye(num_matvecs, num_matvecs) - Q[:, :num_matvecs + 1].T @ Q[:, :num_matvecs + 1])
            ortho_cgs[num_matvecs]["ts"].append(end - start)
            ortho_cgs[num_matvecs]["loss of orthogonality"].append(ortho)
            print(f"CGS time taken: {end - start:.6f} seconds, n={num_matvecs}, orthogonality: {ortho}")

            start = time.time()
            w, Q, H, breakdown = arnoldi(A_scipy, b_scipy / b_length, num_matvecs, reortho=False, mgs=False, proj=True,
                                         trunc=np.inf)
            end = time.time()
            ortho = np.linalg.norm(np.eye(num_matvecs, num_matvecs) - Q[:, :num_matvecs + 1].T @ Q[:, :num_matvecs + 1])
            ortho_cgs_projreortho[num_matvecs]["ts"].append(end - start)
            ortho_cgs_projreortho[num_matvecs]["loss of orthogonality"].append(ortho)
            print(f"CGS proj time taken: {end - start:.6f} seconds, n={num_matvecs}, orthogonality: {ortho}")

            start = time.time()
            w, Q, H, breakdown = arnoldi(A_scipy, b_scipy / b_length, num_matvecs, reortho=True, mgs=False, proj=False,
                                         trunc=np.inf)
            end = time.time()
            ortho = np.linalg.norm(np.eye(num_matvecs, num_matvecs) - Q[:, :num_matvecs + 1].T @ Q[:, :num_matvecs + 1])
            ortho_cgs_reortho[num_matvecs]["ts"].append(end - start)
            ortho_cgs_reortho[num_matvecs]["loss of orthogonality"].append(ortho)
            print(f"CGS reortho time taken: {end - start:.6f} seconds, n={num_matvecs}, orthogonality: {ortho}")

    with open(f"../outputs/cpu_benchmark_{matrix}.json", "w") as f:
        json.dump({"ortho_mgs": ortho_mgs, "ortho_trunc": ortho_trunc, "ortho_cgs": ortho_cgs,
                  "ortho_cgs_projreortho": ortho_cgs_projreortho, "ortho_cgs_reortho": ortho_cgs_reortho}, f)

    names = {
        "ortho_mgs": "MGS",
        "ortho_trunc": "Trunc.",
        "ortho_cgs": "CGS",
        "ortho_cgs_projreortho": "CGS w/ reo.",
        "ortho_cgs_reortho": "CGS w/ proj.",
    }

    data = json.load(open(f"../outputs/cpu_benchmark_{matrix}.json"))
    fig, ax = plt.subplots()
    width = 1/6
    multiplier = 0
    colors = plt.color_sequences["tab10"]
    for n in num_matvecss:
        m2 = 0
        for method in data.keys():
            offset = width * multiplier
            rects = ax.bar(offset, numpy.median(data[method][str(n)]["ts"]), width, label=names[method] if n == 50 else "", color=colors[m2])
            # ax.bar_label(rects, padding=3)
            multiplier += 1
            m2 += 1
        multiplier += 1
    ax.set_ylabel('Runtime [s]')
    ax.set_xticks(np.arange(len(num_matvecss)) + 0.5 - width, num_matvecss)
    ax.set_xlabel("Number of Arnoldi steps")
    ax.legend(loc='upper left', ncols=3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(f"../outputs/cpu_benchmark_{matrix}.pdf")
    plt.show()