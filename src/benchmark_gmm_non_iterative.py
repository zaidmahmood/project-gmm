from sklearn.cluster import KMeans
import numpy as np
import time
import gc


def compute_gmm_non_iterative(X, K=3):
    """"
    Compute GMM-like parameters using K-means clustering.
    """

    # Run K-means
    kmeans = KMeans(n_clusters=K, n_init=1, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # Compute GMM-like parameters (weights, means, variances)
    for k in range(K):
        cluster_points = X[cluster_labels == k]
        _ = len(cluster_points) / len(X)         # πₖ (mixing probability)
        _ = np.mean(cluster_points)              # μₖ (mean)
        _ = np.var(cluster_points)               # σ²ₖ (variance)


def benchmark_gmm_non_iterative(model_name, compute_func, X, runs=10):
    """"
    Benchmark the GMM-like parameter computation.
    """
    times = []

    gc.disable()  # turn off garbage collection

    for _ in range(runs):
        start = time.perf_counter()
        compute_func(X)  # run K-means + parameter computation
        end = time.perf_counter()
        times.append(end - start)

    gc.enable()  # re-enable GC

    # Print benchmark results
    print(f"{model_name}")
    print(f"Median time over {runs} runs: {np.median(times):.6f} s")
    print(f"Mean time: {np.mean(times):.6f} s ± {np.std(times):.6f}")
    return np.mean(times)
