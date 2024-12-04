from venumML.venumpy import small_glwe as vp
import numpy as np

# Precompute the FFT exponential values
def generate_twiddle_factors(n):
    """
    Generates the twiddle factors for the Fast Fourier Transform (FFT) as tuples of real and imaginary parts.

    Parameters
    ----------
    n : int
        Length of the sequence.

    Returns
    -------
    list of tuple
        Twiddle factors represented as tuples of (real, imaginary) values.
    """
    angles = -2 * np.pi * np.arange(n) / n
    return [(np.cos(angle), np.sin(angle)) for angle in angles]

def next_power_of_2(n):
    """
    Calculate the next power of 2 greater than or equal to n.

    Parameters
    ----------
    n : int
        The input integer.

    Returns
    -------
    int
        The smallest power of 2 greater than or equal to n.
    """
    return 1 if n == 0 else 2**(n - 1).bit_length()
    
def pad_ciphervec(ctx, x, pad_length=None,pad_value = 0):
    """
    Pads the input encrypted vector to a specified length using a padding value.

    Parameters
    ----------
    ctx : EncryptionContext
        The encryption context used to encrypt padding values.
    x : np.ndarray
        Encrypted input vector.
    pad_length : int, optional
        Desired length of the padded vector.
    pad_value : int, optional, default=0
        Value to use for padding the vector, which will be encrypted.

    Returns
    -------
    np.ndarray
        Encrypted padded vector.
    """
    N = len(x)
    assert N<=pad_length
    pad_diff = pad_length - N
    pad_vector = np.asarray([ctx.encrypt(pad_value)]*pad_diff)
    padded_x = np.append(x,pad_vector)
    return padded_x
    
def hann_window(ctx,x):
    """
    Applies a Hann window to the input encrypted signal to reduce spectral leakage.

    Parameters
    ----------
    ctx : EncryptionContext
        The encryption context used to encrypt window values.
    x : np.ndarray
        Encrypted input signal.

    Returns
    -------
    np.ndarray
        The input signal multiplied by the encrypted Hann window.
    """
    N = len(x)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N-1))
    window = np.asarray([ctx.encrypt(h) for h in window])
    return x * window

def FFT(ctx,x_unpadded, pad_data = False, window_data = False):
    """
    Computes the Fast Fourier Transform (FFT) of an encrypted signal using the 1D Cooley-Tukey algorithm,
    a recursive implementation of the 1D Cooley-Tukey FFT without using complex numbers.

    Parameters
    ----------
    ctx : EncryptionContext
        The encryption context used to encrypt data and twiddle factors.
    x_unpadded : np.ndarray
        Encrypted input signal.
    pad_data : bool, optional, default=False
        Whether to pad the input data to the next power of 2 for FFT optimisation.
    window_data : bool, optional, default=False
        Whether to apply a Hann window to the input data before the FFT.

    Returns
    -------
    list of tuple
        The FFT of the input signal as a list of tuples (real, imaginary).
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
    """
    Returns the Discrete Fourier Transform sample frequencies for real input signals.

    Parameters
    ----------
    n : int
        Window length.
    d : float, optional, default=1.0
        Sample spacing (inverse of the sampling rate).

    Returns
    -------
    np.ndarray
        Array of frequencies.
    """
    val = 1.0 / (n * d)
    results = np.arange(n // 2 + 1, dtype=int)
    return results * val
