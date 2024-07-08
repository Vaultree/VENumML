# VENumML User Manual

## Requirements

- Supported platforms: MacOS, Linux
- Python 3.10 or newer installed
- VENumpy
- Vaultree License

### VENumpy

You'll need an instance of VENumpy, either as a submodule or wheel file. This library MUST be built prior to running VENumML. Please see directions below to import VENumpy as a wheel file. 

### Vaultree License

You'll need a valid **Vaultree License** to actually use VENumpy. If you don't have a license, or you have an expired one, please contact Vaultree to request one.
In case you already have a valid license, please follow the steps below:

1. Create a new directory called `.vaultree` on your `$HOME` folder.

2. Place the `vaultree.license` file inside the `$HOME/.vaultree`.

## Installation instructions

To install and use VENumpy you need to:

1. Create a Python virtual environment (venv)

```
python3 -m venv <env-folder>
```

2. Activate the venv

```
source <env-foldeer>/bin/activate
```

3. Install VENumpy's **wheel** file on the venv

```
pip install venumpy-0.1.0-cp312-cp312-macosx_11_0_arm64.whl
```

This file name is just an example. The version and platform information may be different.


## VENumML Usage Instructions

To use VENumML demos, we recommend using ipykernel in VScode.

1. Once venumpy is installed in your virtual environment, you are ready to perform calculations over encrypted data!
Navigate to the demos folder and use jupyter notebook within VS Code. You'llalso need ipykernel in your virtual env.

```
pip install ipykernel
```

2. To use jupyter notebook demos (.ipynb), **shift + enter** runs code blocks. This action will prompt you for selecting your virtual env and might prompt installing necessary jupyter extension.

3. In the root directory, make sure you have installed the necessary python dependancies in your venv:

```
pip install -r requirements.txt
```

Congratulations, you're ready to use VENumML with VENumpy! Try out the demos to perform machine learning over encrypted data!


## Debugging - Checking your VENumpy installation

With the same virtual environment from before activated, run the following commands:

1. Start the Python interpreter (make sure your virtual environment is activated).

```
python
```

2. Import `venumpy`

```python
import venumpy
```

If the import prints **no messages**, the installation was successful.

This setup assumes there's a valid Vaultree license on `$HOME/.vaultree`. Check the license section above.

## Usage

To check the src of venumML, navigate to `venumMLlib` to review the underlying data structures. Also check the files in venumpy's `examples` directory to get to know VENumpy's API.

## Licensing

All content accompanying this file is copyrighted to Vaultree.

© 2024 Vaultree. All rights reserved.
