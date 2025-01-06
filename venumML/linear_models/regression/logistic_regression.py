from venumML.venumpy import small_glwe as vp
import numpy as np
from venumML.venum_tools import *

def sigmoid_approximation(ctx, x):
    """@private

    Parameters
    ----------
    ctx : type
        Description of ctx.
    x : type
        Description of x.

    Returns
    -------
    result : type
        Description of result.
    """

    # Define the core logic as an inner function to facilitate vectorization
    def core_sigmoid(ctx, x):
        tc1, tc2, tc3 = ctx.encrypt(0.5), ctx.encrypt(0.25), ctx.encrypt(0.02)
        return tc1 + (tc2 * x) #- ((x * x * x) * tc3)

    # Vectorize the core sigmoid function
    v_core_sigmoid = np.vectorize(core_sigmoid, excluded=['ctx'])

    # Apply it to the input x, which can now be a single value or a NumPy array of encrypted objects
    return v_core_sigmoid(ctx=ctx, x=x)


class EncryptedLogisticRegression:
    """
    Logistic Regression model that supports encrypted computations.

    Attributes
    ----------
    _context : object
        Encryption context to perform encrypted operations.
    _coef_ : numpy.ndarray or None
        Model coefficients (weights).
    _intercept_ : float
        Intercept (bias) term.
    _encrypted_coef_ : numpy.ndarray
        Encrypted model coefficients (weights).
    _encrypted_intercept_ : object
        Encrypted intercept (bias) term.
    """
    
    def __init__(self, ctx):
        """
        Initialise the EncryptedLogisticRegression model.

        Parameters
        ----------
        ctx : object
            Encryption context to handle encrypted operations.
        """

        self._context = ctx
        self._coef_ = None  # Will be initialized based on number of features
        self._intercept_ = 0  # Scalar for the bias term
    
    def fit(self, X, y, num_iterations=1000, learning_rate=0.1):
        """
        Fit the model to the provided data using gradient descent.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix (n_samples, n_features).
        y : numpy.ndarray
            Target vector (n_samples,).
        num_iterations : int, optional
            Number of iterations for gradient descent (default is 1000).
        learning_rate : float, optional
            Learning rate for gradient descent (default is 0.1).
        """
        n_samples, n_features = X.shape
        self._coef_ = np.zeros(n_features)  # Initialize coefficients as a vector
        self._intercept_ = 0  # Initialize intercept (bias)

        for _ in range(num_iterations):
            linear_model = np.dot(X, self._coef_) + self._intercept_  # Linear combination
            y_predicted = 1 / (1 + np.exp(-linear_model))  # Sigmoid function

            # Gradient calculations
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))  # Derivative w.r.t. weights
            db = (1 / n_samples) * np.sum(y_predicted - y)  # Derivative w.r.t. bias

            # Parameter updates
            self._coef_ -= learning_rate * dw
            self._intercept_ -= learning_rate * db

    def encrypted_fit(self, ctx, x, y, lr=0.3, gamma=0.9, epochs=3):
        """
        Fit the model using encrypted data using encrypted data, Nesterov optimization, and a sigmoid approximation.

        Parameters
        ----------
        ctx : object
            Encryption context for handling encrypted operations.
        x : numpy.ndarray
            Encrypted feature matrix (n_samples, n_features).
        y : numpy.ndarray
            Encrypted target vector (n_samples,).
        lr : float, optional
            Learning rate for optimisation (default is 0.3).
        gamma : float, optional
            Momentum term for Nesterov optimisation (default is 0.9).
        epochs : int, optional
            Number of epochs for training (default is 3).
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
        self._encrypted_coef_ = np.atleast_1d(w.squeeze())
        self._encrypted_intercept_ = b

    def encrypt_coefficients(self, ctx):
        """
        Encrypt the model's coefficients and intercept.

        Parameters
        ----------
        ctx : object
            Encryption context to perform encrypted operations.
        """

        self._encrypted_intercept_ = ctx.encrypt(self._intercept_)
        self._encrypted_coef_ = [ctx.encrypt(v) for v in self._coef_]

    def predict(self, encrypted_X, ctx):
        """
        Predict outcomes using encrypted data.

        Parameters
        ----------
        encrypted_X : numpy.ndarray
            Encrypted feature matrix (n_samples, n_features).
        ctx : object
            Encryption context for handling encrypted operations.

        Returns
        -------
        object
            Encrypted predictions.
        """

        # Compute the linear model with encrypted coefficients and intercept
        linear_model = np.dot(encrypted_X, self._encrypted_coef_) + self._encrypted_intercept_

        return linear_model
