![alt text][logo]

[logo]: Vaultree.png "Vaultree Logo"

# VENumML
Encrypted machine learning (EML) library enabled by the Vaultree python library, VENumpy. At its source, VENumpy is built on our core fully homomorphic encryption (FHE) Vaultree Encrypted Number library, VENum.
- [venumMLlib](venumMLlib)
  * Linear Models
    * [Linear Regression](venumMLlib/linear_models/regression/linear_regression.py)
    * [Logistic Regression](venumMLlib/linear_models/regression/logistic_regression.py)
  * Optimization
    * Stochastic gradient descent (SGD)
  * Time Series (Phineus)
    * [FFT](venumMLlib/time_series/Phineus/phineus_FFT.py)
    * [Rolling Average](venumMLlib/time_series/Phineus/phineus_rolling_average.py)
  * Deep Learning
      * [Transformers](venumMLlib/deep_learning/transformer/transformer.py)
      * Facial Recognition (client-server app included in next release)
  * Graphs
      * PageRank (included in next release)
  
- [Demos](demos) 
  * [Linear Regression](demos/linear_regression_demo.ipynb)
  * [Logistic Regression](demos/logistic_regression_demo.ipynb)
  * [Phineus](demos/Phineus_Demo.ipynb)
  * [FFT](demos/phineus_FFT_demo.ipynb)
  * [Rolling Average](demos/phineus_rolling_avg_demo.ipynb)
  * [Facial Recognition](demos/facial_rec_demo/facial_rec_demo.ipynb)
  * [Transformers](demos/transformer_demo.ipynb)

## VENumML: Machine Learning for Encrypted Data

VENumML is a Privacy Preserving Machine Learning (PPML) Python library designed for building and applying machine learning models on encrypted data. It leverages fully homomorphic encryption (FHE) techniques to perform computations on encrypted data without decryption, ensuring data privacy throughout the machine learning workflow.

### **``Getting Started:``**
[Setup Instructions](docs/venumML-user-manual.md)


### **``Overview``**

VENumML is an open-source library designed to leverage Fully Homomorphic Encryption (FHE) for building and applying machine learning models on encrypted data. While VENumML itself is open-source, it currently relies on Vaultree's internal FHE library, VENumpy, to perform the necessary encrypted computations.

We believe VENumML's open-source architecture offers a valuable framework for developing secure and privacy-preserving machine learning applications. By making VENumML open-source, we aim to:

* Foster collaboration and innovation in the field of encrypted machine learning.
* Encourage developers to explore the potential of VENumML for various use cases.
* Showcase the capabilities of VENumML in conjunction with VENumpy.
* Provide transparency: Users can understand the core functionalities and architecture of VENumML.
* Build community: Developers can contribute to the project's development and improve its functionalities.

#### Using VENumML with VENumpy:

To currently run VENumML and experience its functionalities, you will need a license for Vaultree's VENumpy library. If you're interested in obtaining a license for VENumpy and exploring VENumML's capabilities, please [contact Vaultree](https://customer.support.vaultree.com/).



### **``Key Features:``**

* **Encrypted Machine Learning:** Implement various machine learning models while keeping the underlying data encrypted.
* **Homomorphic Encryption Support:** Works with Vaultree's VENumpy library that provides FHE functionalities.
* **Privacy-Preserving Predictions:** Make predictions on encrypted data without revealing the original features.

**Current Components:**

This library is under development and currently includes implementations for:

**Regression:**

* Linear Regression: Train and use a linear regression model on encrypted data for continuous target variables.
* Logistic Regression: Train and use a logistic regression model on encrypted data for binary classification tasks.

**Deep Learning (Under Development):**

* Transformer: Explore encrypted implementations of the Transformer architecture for various deep learning tasks.

**Time Series (Phineus Sub-package):**

* FFT (Fast Fourier Transform): Perform FFT on encrypted time series data to analyze frequency domain information while preserving privacy.
* Rolling Average: Calculate rolling averages on encrypted time series data for smoothing and trend analysis.
 


### **``Contributing:``**

We welcome contributions to this open-source library. If you're interested in helping develop or improve VENumML, please refer to the contribution guidelines (to be added in a separate document).

### **``Disclaimer:``**

This library is under active development and might contain limitations or bugs. Use it for research or experimentation purposes with caution.

## Licensing

All content accompanying this file is copyrighted to Vaultree.

© 2024 Vaultree. All rights reserved.

