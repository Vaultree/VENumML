import venumpy
import numpy as np
from venumMLlib.venum_tools import encrypt_array


def rolling_average(ctx, values, window):
 
    rolling_avg = encrypt_array(np.zeros(len(values)),ctx)
    
    for i in range(len(values)):
        if i < window - 1:
            rolling_avg[i] = np.mean(values[:i + 1])
        else:
            rolling_avg[i] = np.mean(values[i - window + 1:i + 1])
    
    return rolling_avg
