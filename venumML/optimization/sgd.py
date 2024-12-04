import numpy as np
from venumML.venumpy import small_glwe as vp
from venumML.venum_tools import *

class Nesterov:
    """
    A class implementing Nesterov's Accelerated Gradient Descent (NAGD) for encrypted data.
    This class leverages homomorphic encryption to securely compute model parameters 
    without decrypting sensitive data.

    Attributes
    ----------
    context : EncryptionContext
        The encryption context that provides encryption and decryption methods.
    lr : float
        Learning rate for gradient descent.
    gamma : float
        Momentum factor for Nesterov's accelerated gradient descent.
    epochs : int
        Number of epochs to run the optimisation.
    """

    def __init__(self, ctx, lr=0.3, gamma=0.9, epochs=10):
        """
        Initialises the Nesterov optimiser with an encryption context and optimisation hyperparameters.

        Parameters
        ----------
        ctx : EncryptionContext
            Encryption context used to encrypt values and perform secure computations.
        lr : float, optional, default=0.3
            Learning rate for gradient descent.
        gamma : float, optional, default=0.9
            Momentum factor for Nesterov's accelerated gradient descent.
        epochs : int, optional, default=10
            Number of epochs to perform the optimisation.
        """

        self._context = ctx
        self._lr = lr
        self._gamma = gamma
        self._epochs = epochs

    def venum_nesterov_agd(self,ctx, x, y):
        """
        Applies Nesterov's Accelerated Gradient Descent on encrypted data to optimise weights.

        Parameters
        ----------
        ctx : EncryptionContext
            The encryption context used to encrypt values.
        x : encrypted array-like, shape (n_samples, n_features)
            Encrypted input data.
        y : encrypted array-like, shape (n_samples, 1)
            Encrypted target values.

        Returns
        -------
        encrypted_intercept : encrypted float
            The encrypted intercept term after optimisation.
        encrypted_coef : encrypted array-like, shape (n_features,)
            The encrypted coefficient(s) after optimisation.
        losses : list of float
            List of loss values recorded at each epoch.

        Notes
        -----
        This method initialises the model's parameters with random values, encrypts them, 
        and then iteratively updates them using Nesterov's accelerated gradient descent.
        """

        if x.ndim == 1:
            x = x.reshape(-1, 1)  # Ensuring x is at least 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)  # Ensuring y is at least 2D
        
        n_samples, n_features = x.shape        
        # Initialize weights and biases
        w = np.random.randn(n_features, 1) * np.sqrt(1 / n_features)
        b = np.random.randn()
        
        # Encrypt weights, biases, and hyperparameters
        w = encrypt_array(w, ctx)
        b = ctx.encrypt(b)
        lr = ctx.encrypt(self._lr)
        gamma = ctx.encrypt(self._gamma)
    
        velocity_w = encrypt_array(np.zeros((n_features, 1)), ctx)
        velocity_b = ctx.encrypt(0)
        
        losses = []
        
        for i in range(self._epochs):
            # Look-ahead weights and bias
            w_look_ahead = w - velocity_w * gamma
            b_look_ahead = b - velocity_b * gamma
            
            # Predict with look-ahead parameters
            y_pred = x @ w_look_ahead + b_look_ahead  
            
            # Compute error and loss
            error = y_pred - y
            loss = np.mean(error**2)
            losses.append(loss)
            
            # Compute gradients
            grad_w = np.mean(x * error, axis=0, keepdims=True).T
            grad_b = np.mean(error)
            
            # Update velocities
            velocity_w = velocity_w * gamma + grad_w * lr
            velocity_b = velocity_b * gamma + grad_b * lr

            # Update parameters
            w = w - velocity_w
            b = b - velocity_b
        
        encrypted_intercept = b_look_ahead
        encrypted_coef = np.atleast_1d(w_look_ahead.squeeze())
        
        return encrypted_intercept, encrypted_coef, losses