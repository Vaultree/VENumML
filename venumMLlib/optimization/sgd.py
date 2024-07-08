import numpy as np
import venumpy
from venumMLlib.venum_tools import *
class Nesterov:
    def __init__(self, ctx, lr=0.3, gamma=0.9, epochs=10):
        self.context = ctx
        self.lr = lr
        self.gamma = gamma
        self.epochs = epochs

    def venum_nesterov_agd(self,ctx, x, y):
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
        lr = ctx.encrypt(self.lr)
        gamma = ctx.encrypt(self.gamma)
    
        velocity_w = encrypt_array(np.zeros((n_features, 1)), ctx)
        velocity_b = ctx.encrypt(0)
        
        losses = []
        
        for i in range(self.epochs):
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
