from venumML.venumpy import small_glwe as vp
import numpy as np
from venumML.venum_tools import *

def sigmoid_approximation(ctx, x):
    # Define the core logic as an inner function to facilitate vectorization
    def core_sigmoid(ctx, x):
        tc1, tc2, tc3 = ctx.encrypt(0.5), ctx.encrypt(0.25), ctx.encrypt(0.02)
        return tc1 + (tc2 * x) #- ((x * x * x) * tc3)

    # Vectorize the core sigmoid function
    v_core_sigmoid = np.vectorize(core_sigmoid, excluded=['ctx'])

    # Apply it to the input x, which can now be a single value or a NumPy array of encrypted objects
    return v_core_sigmoid(ctx=ctx, x=x)


class EncryptedLogisticRegression:
    
    def __init__(self, ctx):
        self.context = ctx
        self.coef_ = None  # Will be initialized based on number of features
        self.intercept_ = 0  # Scalar for the bias term
    
    def fit(self, X, y, num_iterations=1000, learning_rate=0.1):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)  # Initialize coefficients as a vector
        self.intercept_ = 0  # Initialize intercept (bias)

        for _ in range(num_iterations):
            linear_model = np.dot(X, self.coef_) + self.intercept_  # Linear combination
            y_predicted = 1 / (1 + np.exp(-linear_model))  # Sigmoid function

            # Gradient calculations
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))  # Derivative w.r.t. weights
            db = (1 / n_samples) * np.sum(y_predicted - y)  # Derivative w.r.t. bias

            # Parameter updates
            self.coef_ -= learning_rate * dw
            self.intercept_ -= learning_rate * db

    def encrypted_fit(self, ctx, x, y, lr=0.3, gamma=0.9, epochs=3):
        """
        Fits the logistic regression model using encrypted data, Nesterov optimization, 
        and a sigmoid approximation.
        """
        encrypted_gamma = ctx.encrypt(gamma)
        encrypted_lr = ctx.encrypt(lr)

        if x.ndim == 1:
            x = x.reshape(-1, 1)  # Ensure x is at least 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)  # Ensure y is at least 2D

        n_samples, n_features = x.shape

        # Initialize encrypted weights and intercept
        w = encrypt_array(6e-2*(np.random.randn(n_features, 1) * np.sqrt(1 / n_features)), ctx)
        b = ctx.encrypt(np.random.randn())

        velocity_w = encrypt_array(np.zeros((n_features, 1)), ctx)
        velocity_b = ctx.encrypt(0)
        # Training loop with Nesterov optimization
        for _ in range(epochs):
            # Look-ahead weights and bias
            w_look_ahead = w - velocity_w * encrypted_gamma
            b_look_ahead = b - velocity_b * encrypted_gamma

            # Linear model and sigmoid approximation
            linear_model = np.dot(x, w_look_ahead) + b_look_ahead
            y_pred = sigmoid_approximation(ctx, linear_model)  # Encrypted sigmoid

            # Compute error
            error = y_pred - y

            # Compute gradients (encrypted)
            grad_w = np.mean(x * error, axis=0, keepdims=True).T
            grad_b = np.mean(error)


            # Update velocities
            # print(velocity_w,grad_w,encrypted_lr,gamma)
            velocity_w = velocity_w * encrypted_gamma + grad_w * encrypted_lr
            velocity_b = velocity_b * encrypted_gamma + grad_b * encrypted_lr

            # Update weights and bias
            w = w - velocity_w
            b = b - velocity_b

        # Store encrypted parameters
        self.encrypted_coef_ = np.atleast_1d(w.squeeze())
        self.encrypted_intercept_ = b

    def encrypt_coefficients(self, ctx):
        self.encrypted_intercept_ = ctx.encrypt(self.intercept_)
        self.encrypted_coef_ = [ctx.encrypt(v) for v in self.coef_]

    def predict(self, encrypted_X, ctx):
        # Compute the linear model with encrypted coefficients and intercept
        linear_model = np.dot(encrypted_X, self.encrypted_coef_) + self.encrypted_intercept_

        return linear_model
