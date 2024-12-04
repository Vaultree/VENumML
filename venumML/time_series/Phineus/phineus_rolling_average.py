from venumML.venumpy import small_glwe as vp
import numpy as np
from venumML.venum_tools import encrypt_array


def rolling_average(ctx, values, window):
    """
    Computes the rolling average of an encrypted array with a specified window size.

    Parameters
    ----------
    ctx : EncryptionContext
        The encryption context used to encrypt the initial rolling average array.
    values : np.ndarray
        Encrypted array of input values.
    window : int
        The size of the window over which to compute the rolling average.

    Returns
    -------
    np.ndarray
        Encrypted array containing the rolling averages for each position in the input array.

    Notes
    -----
    For the first few elements where the window size is not fully available, the function 
    calculates the average over the available elements up to the current position.
    """
 
    rolling_avg = encrypt_array(np.zeros(len(values)),ctx)
    
    for i in range(len(values)):
        if i < window - 1:
            rolling_avg[i] = np.mean(values[:i + 1])
        else:
            rolling_avg[i] = np.mean(values[i - window + 1:i + 1])
    
    return rolling_avg
