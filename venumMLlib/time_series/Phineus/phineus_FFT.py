import venumpy
import numpy as np

# Precompute the FFT exponential values
def generate_twiddle_factors(N):
    """
    Generate the twiddle factors for FFT, represented as tuples of real and imaginary parts.
    
    Parameters:
    N (int): Length of the sequence.
    
    Returns:
    list: Twiddle factors as tuples of (real, imaginary).
    """
    angles = -2 * np.pi * np.arange(N) / N
    return [(np.cos(angle), np.sin(angle)) for angle in angles]

def next_power_of_2(n):
    return 1 if n == 0 else 2**(n - 1).bit_length()
    
def pad_ciphervec(ctx, x,pad_length=None,pad_value = 0):
    N = len(x)
    assert N<=pad_length
    pad_diff = pad_length - N
    pad_vector = np.asarray([ctx.encrypt(pad_value)]*pad_diff)
    padded_x = np.append(x,pad_vector)
    return padded_x
    
def hann_window(ctx,x):
    """Apply a Hann window to the input signal."""
    N = len(x)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N-1))
    window = np.asarray([ctx.encrypt(h) for h in window])
    return x * window

def FFT(ctx,x_unpadded, pad_data = False, window_data = False):
    """
    A recursive implementation of the 1D Cooley-Tukey FFT without using complex numbers.
    
    Parameters:
    x (np.ndarray): Input signal.
    
    Returns:
    np.ndarray: FFT of the input signal as tuples of (real, imaginary).
    """
    N = len(x_unpadded)
    if N == 1:
        return [(x_unpadded[0], ctx.encrypt(0))]
    else:
        if window_data is True:
            x_unpadded = hann_window(ctx,x_unpadded)
        else:
            pass
        if pad_data:
            padded_length = next_power_of_2(N)  
            x = pad_ciphervec(ctx, x_unpadded, padded_length)
            N = padded_length
        else:
            x = x_unpadded
        X_even = FFT(ctx, x[::2])
        X_odd = FFT(ctx, x[1::2])
        factor = generate_twiddle_factors(N)
        factor = [(ctx.encrypt(c[0]),ctx.encrypt(c[1])) for c in factor]
        X = [(ctx.encrypt(0), ctx.encrypt(0))] * N
        for k in range(N // 2):
            twiddle_real, twiddle_imag = factor[k]
            odd_real, odd_imag = X_odd[k]
            # Calculate twiddle factor times odd part (real and imaginary separately)
            twiddle_odd_real = twiddle_real * odd_real - twiddle_imag * odd_imag
            twiddle_odd_imag = twiddle_real * odd_imag + twiddle_imag * odd_real
            # Combine even and twiddle*odd parts
            X[k] = (X_even[k][0] + twiddle_odd_real, X_even[k][1] + twiddle_odd_imag)
            X[k + N // 2] = (X_even[k][0] - twiddle_odd_real, X_even[k][1] - twiddle_odd_imag)
        return X
        
def rfftfreq(n, d=1.0):
    val = 1.0 / (n * d)
    results = np.arange(n // 2 + 1, dtype=int)
    return results * val
