import numpy as np
import matplotlib.pyplot as plt


def plot_gmm(model, data, bins=30):
    """
    Plots the Gaussian Mixture Model (GMM) probability density function (PDF) 
    overlaid on a histogram of the given data.

    Args:
        model (sklearn.mixture.GaussianMixture): The trained GMM model used to 
            compute the PDF.
        data (array-like): The input data to be plotted as a histogram.
        bins (int, optional): The number of bins to use for the histogram. 
            Defaults to 30.

    Returns:
        None: This function displays the plot but does not return any value.

    Notes:
        - The x-axis represents the load (MW).
        - The y-axis represents the density.
        - The function uses matplotlib to generate the plot.
    """
    x = np.linspace(min(data), max(data), 1000).reshape(-1, 1)
    logprob = model.score_samples(x)
    pdf = np.exp(logprob)
    plt.hist(data, bins=bins, density=True, alpha=0.5, label="Data Histogram")
    plt.plot(x, pdf, label="GMM PDF", linewidth=2)
    plt.xlabel("Load (MW)")
    plt.ylabel("Density")
    plt.title("GMM Fit to Load Data")
    plt.legend()
    plt.grid(True)
    plt.show()
