import venumpy
import numpy as np
import math

class EncryptedLogisticRegression:
    
    def __init__(self, ctx):
        self.context = ctx
        self.coef_ = 0  
        self.intercept_ = 0  
    
    def fit(self, X, y, num_iterations=1000,learning_rate = 0.1):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = 1 / (1 + np.exp(-linear_model))

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.coef_ -= learning_rate * dw
            self.intercept_ -= learning_rate * db

    def encrypt_coefficients(self, ctx):
        self.encrypted_intercept_ = ctx.encrypt(self.intercept_)
        self.encrypted_coef_ = [ctx.encrypt(v) for v in self.coef_]

    def predict(self, encrypted_X, ctx):

        linear_model = np.dot(encrypted_X, self.encrypted_coef_) + self.encrypted_intercept_

        return linear_model

   