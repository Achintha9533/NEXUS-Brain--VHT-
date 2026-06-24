import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.multioutput import MultiOutputRegressor
import torch
import torch.nn as nn

class StreamingResidualGP:
    def __init__(self, kernel, alpha=0.5, random_state=42, retrain_every=1):
        self.kernel = kernel
        self.alpha = alpha
        self.random_state = random_state
        self.retrain_every = retrain_every
        self.X_hist = []
        self.Y_hist = []
        self.n_updates = 0
        self.model = MultiOutputRegressor(
            GaussianProcessRegressor(
                kernel=self.kernel,
                alpha=self.alpha,
                random_state=self.random_state
            )
        )

    def _clean_and_flatten(self, arr):
        """Helper to ensure inputs are flat, finite float matrices."""
        a = np.asarray(arr, dtype=float).reshape(-1)
        # Handle structural missingness or segmentation failures gracefully
        if not np.all(np.isfinite(a)):
            a = np.nan_to_num(a, nan=0.0, posinf=1e6, neginf=-1e6)
        return a

    def fit_initial(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if len(X) < 1 or len(Y) < 1:
            raise ValueError("Cannot fit GP: empty training data.")
        
        self.X_hist = [self._clean_and_flatten(x) for x in X]
        self.Y_hist = [self._clean_and_flatten(y) for y in Y]
        self.model.fit(np.asarray(self.X_hist), np.asarray(self.Y_hist))
        return self

    def update(self, x_new, y_new=None):
        self.X_hist.append(self._clean_and_flatten(x_new))
        if y_new is not None:
            self.Y_hist.append(self._clean_and_flatten(y_new))
            self.n_updates += 1
            if self.n_updates % self.retrain_every == 0 and len(self.Y_hist) >= 2:
                self.model.fit(np.asarray(self.X_hist), np.asarray(self.Y_hist))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X)

    def predict_with_uncertainty(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        mean = self.model.predict(X)
        stds = []
        for est in self.model.estimators_:
            _, std = est.predict(X, return_std=True)
            stds.append(std)
        return mean, np.vstack(stds).T