import venumpy
import numpy as np
from venumMLlib.venum_tools import *
from venumMLlib.optimization.sgd import Nesterov

class EncryptedLinearRegression:
    
    def __init__(self, ctx):
        self.context = ctx
        self.coef_ = None
        self.intercept_ = None
        self.encrypted_intercept_ = ctx.encrypt(0)
        self.encrypted_coef_ = ctx.encrypt(0)

    
    def encrypted_fit(self, ctx, x, y, lr=0.3, gamma=0.9, epochs=10):
        optimizer = Nesterov(ctx)
        encrypted_intercept, encrypted_coef, losses = optimizer.venum_nesterov_agd(ctx,x,y)
        
        self.encrypted_intercept_ = encrypted_intercept
        self.encrypted_coef_ = encrypted_coef

    
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept_ = theta_best[0]
        self.coef_ = theta_best[1:]

    def encrypt_coefficients(self, ctx):
        self.encrypted_intercept_ = ctx.encrypt(self.intercept_)
        self.encrypted_coef_ = [ctx.encrypt(v) for v in self.coef_]

    def predict(self, encrypted_X, ctx):
        


        encrypted_prediction = encrypted_X @ self.encrypted_coef_ + self.encrypted_intercept_

        return encrypted_prediction
