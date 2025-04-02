import numpy as np
import time
import gc
from sklearn.mixture import GaussianMixture


def benchmark_gmm(model_name, gmm_obj, X, runs=10):
    """
    Benchmark the GMM fitting process.
    """

    times = []
    iterations = []

    gc.disable()  # prevent garbage collection during benchmark

    for _ in range(runs):
        gmm = GaussianMixture(**gmm_obj.get_params())  # fresh model each time
        start = time.perf_counter()
        gmm.fit(X)
        end = time.perf_counter()

        times.append(end - start)
        iterations.append(gmm.n_iter_)

    gc.enable()

    # Report stats
    print(f"{model_name}")
    print(f"Median time over {runs} runs: {np.median(times):.6f} s")
    print(f"Mean time: {np.mean(times):.6f} s ± {np.std(times):.6f}")
    print(
        f"Average EM iterations: {np.mean(iterations):.2f} ± {np.std(iterations):.2f}")

    return np.mean(times), np.mean(iterations)
