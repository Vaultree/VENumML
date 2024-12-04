import numpy as np
from venumML.venumpy import small_glwe as vp

def encrypt_array(array, ctx):
    """
    Recursively encrypts every value in an n-dimensional numpy array.

    Parameters
    ----------
    array : np.ndarray
        The n-dimensional numpy array to encrypt.
    ctx : EncryptionContext
        Encryption context providing an encrypt method.

    Returns
    -------
    np.ndarray
        An n-dimensional numpy array with all values encrypted.

    Raises
    ------
    ValueError
        If the input is not a numpy array.
    """
    # Check if the array is not a numpy array
    if not isinstance(array, np.ndarray):
        raise ValueError("Input must be a numpy array")

    # Function to recursively encrypt elements
    def encrypt_element(element):
        # If the element is an array, apply encryption recursively
        if isinstance(element, np.ndarray):
            return np.array([encrypt_element(sub_element) for sub_element in element])
        else:
            # Encrypt the single element
            return ctx.encrypt(element)

    # Apply encryption to each element in the array
    encrypted_array = np.array([encrypt_element(sub_array) for sub_array in array])

    return encrypted_array

def decrypt_array(array):
    """
    Recursively decrypts every value in an n-dimensional numpy array of encrypted objects.

    Parameters
    ----------
    array : np.ndarray
        The n-dimensional numpy array containing encrypted objects to decrypt.

    Returns
    -------
    np.ndarray
        An n-dimensional numpy array with all values decrypted.

    Raises
    ------
    ValueError
        If the input is not a numpy array.
    """

    # Check if the array is not a numpy array
    if not isinstance(array, np.ndarray):
        raise ValueError("Input must be a numpy array")

    # Function to recursively decrypt elements
    def decrypt_element(element):
        # If the element is an array, apply decryption recursively
        if isinstance(element, np.ndarray):
            return np.array([decrypt_element(sub_element) for sub_element in element])
        else:
            # Decrypt the single element
            return element.decrypt()

    # Apply decryption to each element in the array
    decrypted_array = np.array([decrypt_element(sub_array) for sub_array in array])

    return decrypted_array
