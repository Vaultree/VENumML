<div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/Vaultree/venumML/main/Vaultree_EML.png" alt="Logo" style="width: 100%;">
</div>

`VENumML` is a Privacy Preserving Machine Learning (PPML) library designed for building and applying machine learning models on encrypted data. With Vaultree's `VENumpy` library, `VENumML` leverages fully homomorphic encryption (FHE) techniques to perform computations on encrypted data without decryption, ensuring data privacy throughout the machine learning workflow. This repo is available to install via [PyPI](https://pypi.org/project/venumML/), see the installation instructions below for further details.

Explore the `VENumML` [Documentation](https://docs.eml.vaultree.com) to learn more about our tool. Visit our [GitHub Repository](https://www.github.com/Vaultree/venumML) to access the codebase or check out the [demos](https://github.com/Vaultree/venumML/tree/main/demos) showcasing the capabilities of `VENumML`.

## **`VENumML` Key Features**

* **Encrypted Machine Learning:** Implement various machine learning models while keeping the underlying data encrypted.
* **Homomorphic Encryption Support:** Works with Vaultree's `VENumpy` library that provides FHE functionalities.
* **Privacy-Preserving Predictions:** Make predictions on encrypted data without revealing the original features.

## **Modules**

The `VENumML` library is under active development and currently includes implementations for:

* Linear Models
    * [Linear Regression](https://docs.eml.vaultree.com/venumML/linear_models/regression/linear_regression.html): Train and use a linear regression model on encrypted data for continuous target variables.
    * [Logistic Regression](https://docs.eml.vaultree.com/venumML/linear_models/regression/logistic_regression.html): A logistic regression model on encrypted data for binary classification tasks.
* Optimization
    * [Stochastic gradient descent (SGD)](https://docs.eml.vaultree.com/venumML/optimization/sgd.html): An implementation of Nesterov's Accelerated Gradient Descent on encrypted data.
* Time Series (Phineus)
    * [Fast Fourier Transform](https://docs.eml.vaultree.com/venumML/time_series/Phineus/phineus_FFT.html): Perform FFT on encrypted time series data to analyze frequency domain information while preserving privacy.
    * [Time Series](https://docs.eml.vaultree.com/venumML/time_series/Phineus/phineus_timeseries.html): Calculate rolling averages on encrypted time series data for smoothing and trend analysis.
* Deep Learning
    * [Transformers](https://docs.eml.vaultree.com/venumML/deep_learning/transformer/transformer.html): Explore encrypted implementations of the Transformer architecture for various deep learning tasks.
* Graphs
    * [Venum Graphs](https://docs.eml.vaultree.com/venumML/graphs/venum_graph.html): Create, or import from a `pandas` dataframe or `NetworkX` object to an encrypted node-edge graph data structure and a built-in PageRank score function.
* Approximation Functions
    * [Approximation Functions](https://docs.eml.vaultree.com/venumML/approx_functions.html): Approximation functions for softmax, tanh and sigmoid activation functions.

## **Installation**

The current version of `VENumML` supports Python 3.10/3.11 running on MacOS or Linux. It is recommeneded to install `VENumML` in a virtual environment.

```bash
python -m venv <env_name>
```

At installation, `pip install` will automatically select the correct version of `VENumML` for your platform:

```python
pip install venumML
```

**Manual Installation**

For manual installation, we have the following wheels available for various systems on our [Github Repository](https://github.com/Vaultree/venumML).

If you prefer to install manually, pre-built wheel files are available for the following platforms:

* Linux: venumML-x.x.x-manylinux_2_31_x86_64.whl
* MacOS (Intel): venumML-x.x.x-macosx_x86_64.whl
* MacOS (ARM): venumML-x.x.x-macosx_arm64.whl

```bash
pip install /path/to/venumML-x.x.x-<platform>.whl
```

where `<platform>` should be the appropriate identifier (`manylinux`, `macosx_x86_64`, or `macosx_arm64`).

## **`VENumpy`**

Vaultree Encrypted Numbers, `VENumpy`, the Python FHE (Fully Homomorphic Encryption) library developed by Vaultree provides the underlying technology of `VENumML`.

## **Demos**

For sample usage of `VENumML`, please see the notebook demos on our [Github repository](https://github.com/Vaultree/venumML/tree/main/demos).

## **Support and Contribution**

Please read our [Docs](https://docs.eml.vaultree.com/).

For other support, bug reports, feature requests or contribution to the `VENumML` project, please contact us via our [Support Portal](https://customer.support.vaultree.com/servicedesk/customer/portals).

## **FAQ**

1. **What scheme is this based on?**
   
   VENumML is backed by VENum (Vaultree Encryption Numbers) which is a Rust production-grade homomorphic encryption library.
   We are open-sourcing a Python implementation of the latest version of the library that you can check out at [venum-python-backend](https://github.com/Vaultree/venum-python-backend).
   This is a simplified version which will continue to update so that it is at functional parity with our Rust library. Additionally the VenumML library will be updated as compatible features come online.

2. **Can I use this project with Docker?**

    Yes, provided that the Docker environments match the requirements for `venumML`.

3. **How do I build or package the project?**

    `venumML` is built on top of our `VENumpy` library using our proprietary FHE technology, which is not publicly available at the moment.

4. **Can I use this library in a production environment?**

    See [Note](#Note).

## **Note**

This is the community edition of our EML product. It is free to use under the BSD 3-Clause Clear license only for research, prototyping, and experimentation purposes, not for production purposes. For any commercial use of Vaultree’s open source code, companies must purchase our commercial patent license. By installing this product, you agree that Vaultree will not be held liable for any adverse impact, whether due to commercial deployment or otherwise. We pledge to update this edition as we continue to develop our product. Stay tuned for future changes to our support for production use of this edition and we welcome your contributions to make change happen!

## **Licence**

This project is licensed under the BSD 3-Clause License. See the [LICENSE](https://github.com/Vaultree/venumML/blob/main/LICENSE) file for details.
