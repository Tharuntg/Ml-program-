# Program 6: Locally Weighted Regression (non-parametric)
# NOTE: this program was not in the uploaded PDF; written from the standard lab syllabus.
import numpy as np
import matplotlib.pyplot as plt


def gaussian_weights(x_query, X, tau):
    """Weight each training point by its closeness to the query point."""
    return np.exp(-((X[:, 1] - x_query[1]) ** 2) / (2 * tau ** 2))


def locally_weighted_regression(x_query, X, y, tau):
    """Predict y at x_query by solving a weighted least-squares problem."""
    w = gaussian_weights(x_query, X, tau)
    W = np.diag(w)
    theta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)
    return x_query @ theta


# Synthetic data: noisy sine curve
np.random.seed(42)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + 0.1 * np.random.randn(100)

# Add bias column
X = np.c_[np.ones(len(x)), x]

tau_values = [0.1, 0.3, 0.5, 1.0]
x_test = np.linspace(0, 2 * np.pi, 200)
X_test = np.c_[np.ones(len(x_test)), x_test]

plt.figure(figsize=(14, 10))
for i, tau in enumerate(tau_values, 1):
    y_pred = np.array([locally_weighted_regression(xq, X, y, tau) for xq in X_test])
    plt.subplot(2, 2, i)
    plt.scatter(x, y, c="blue", s=15, label="Training data")
    plt.plot(x_test, y_pred, c="red", label=f"LWR (tau={tau})")
    plt.title(f"Locally Weighted Regression, tau = {tau}")
    plt.legend()

plt.tight_layout()
plt.show()
