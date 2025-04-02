from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import norm


def compute_rmse(bin_centers, hist_vals, weights, means, stds):
    """
    Compute the RMSE between the histogram and the predicted PDF from the GMM.
    """

    # Predicted PDF from GMM
    pdf_pred = np.zeros_like(bin_centers)
    for i in range(len(weights)):
        pdf_pred += weights[i] * norm.pdf(bin_centers, means[i], stds[i])

    # RMSE between histogram and predicted PDF
    rmse = np.sqrt(mean_squared_error(hist_vals, pdf_pred))
    return rmse
