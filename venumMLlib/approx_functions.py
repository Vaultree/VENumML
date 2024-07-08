import numpy as np
import venumpy

def softmax_approximation(ctx, x, D):
    # Define the core logic as an inner function to facilitate vectorization
    def core_softmax(ctx, x, D):
        tc1, tc2, tc3 = ctx.encrypt(1.0), ctx.encrypt(.5), ctx.encrypt(.17)
        return (tc1 + x + tc2*(x*x) + tc3*(x*x*x)) / D

    # Vectorize the core softmax function
    v_core_softmax = np.vectorize(core_softmax, excluded=['ctx', 'D'])

    # Apply it to the input x, which can now be a single value or a NumPy array of encrypted objects
    return v_core_softmax(ctx=ctx, x=x, D=D)

def tanh_approximation(ctx, x):
    taylor_constant = self.ctx.encrypt(0.333)
    return x - taylor_constant * (x * x * x)


def sigmoid_approximation(ctx, x):
    # Define the core logic as an inner function to facilitate vectorization
    def core_sigmoid(ctx, x):
        tc1, tc2, tc3 = ctx.encrypt(0.5), ctx.encrypt(0.25), ctx.encrypt(0.02)
        return tc1 + (tc2 * x) - ((x * x * x) * tc3)

    # Vectorize the core sigmoid function
    v_core_sigmoid = np.vectorize(core_sigmoid, excluded=['ctx'])

    # Apply it to the input x, which can now be a single value or a NumPy array of encrypted objects
    return v_core_sigmoid(ctx=ctx, x=x)

