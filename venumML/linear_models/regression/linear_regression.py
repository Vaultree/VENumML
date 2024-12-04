from venumML.venumpy import small_glwe as vp
import numpy as np
from venumML.venum_tools import *
from venumML.optimization.sgd import Nesterov

class EncryptedLinearRegression:
    """
    A linear regression model that supports encrypted training and prediction.

    Attributes
    ----------
    context : EncryptionContext
        The encryption context that provides encryption and decryption methods.
    coef_ : array-like, shape (n_features,)
        Coefficients of the linear model after fitting (in plaintext).
    intercept_ : float
        Intercept of the linear model after fitting (in plaintext).
    encrypted_intercept_ : encrypted float
        Encrypted intercept of the model, used in encrypted prediction.
    encrypted_coef_ : list of encrypted floats
        Encrypted coefficients of the model, used in encrypted prediction.
    """
    
    def __init__(self, ctx):
        """
        Initialises the EncryptedLinearRegression model with a given encryption context.

        Parameters
        ----------
        ctx : EncryptionContext
            The encryption context used to encrypt values.
        """

        self._context = ctx
        self._coef_ = None
        self._intercept_ = None
        self._encrypted_intercept_ = ctx.encrypt(0)
        self._encrypted_coef_ = ctx.encrypt(0)

    
    def encrypted_fit(self, ctx, x, y, lr=0.3, gamma=0.9, epochs=10):
        """
        Fits the linear regression model on encrypted data using Nesterov's accelerated gradient descent.

        Parameters
        ----------
        ctx : EncryptionContext
            The encryption context used to encrypt and decrypt values.
        x : encrypted array-like, shape (n_samples, n_features)
            Encrypted input data.
        y : encrypted array-like, shape (n_samples,)
            Encrypted target values.
        lr : float, optional, default=0.3
            Learning rate for the optimizer.
        gamma : float, optional, default=0.9
            Momentum parameter for Nesterov's accelerated gradient descent.
        epochs : int, optional, default=10
            Number of epochs to run for optimization.
        """

        optimizer = Nesterov(ctx)
        encrypted_intercept, encrypted_coef, losses = optimizer.venum_nesterov_agd(ctx,x,y)
        
        self._encrypted_intercept_ = encrypted_intercept
        self._encrypted_coef_ = encrypted_coef

    
    def fit(self, X, y):
        """
        Fits the linear regression model using ordinary least squares.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Plaintext input data.
        y : array-like, shape (n_samples,)
            Plaintext target values.
        """

        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self._intercept_ = theta_best[0]
        self._coef_ = theta_best[1:]

    def encrypt_coefficients(self, ctx):
        """
        Encrypts the model's coefficients and intercept after fitting.

        Parameters
        ----------
        ctx : EncryptionContext
            The encryption context used to encrypt plaintexts.
        """

        self._encrypted_intercept_ = ctx.encrypt(self._intercept_)
        self._encrypted_coef_ = [ctx.encrypt(v) for v in self._coef_]

    def predict(self, encrypted_X, ctx):
        """
        Predicts outcomes using encrypted input data and the model's encrypted coefficients.

        Parameters
        ----------
        encrypted_X : encrypted array-like, shape (n_samples, n_features)
            Encrypted input data for making predictions.
        ctx : EncryptionContext
            The encryption context used to encrypt and decrypt values.

        Returns
        -------
        encrypted_prediction : encrypted array-like, shape (n_samples,)
            The encrypted predictions based on the encrypted model coefficients and intercept.
        """
        
        encrypted_prediction = encrypted_X @ self._encrypted_coef_ + self._encrypted_intercept_
        return encrypted_prediction
