from setuptools import setup, find_packages

setup(
    name='venumML',
    version='0.2.1',
    include_package_data=True,
    package_data={
        'venumML.venumpy': ['venumpy.abi3.so'],
    },
    packages=find_packages(),
    install_requires=[
        # TODO: specify package versions to avoid breakage
        'scipy',
        'deepface',
        'tensorflow',
        'matplotlib',
        'numpy',
        'scikit-learn',
        'transformers',
        'tqdm',
        'torch',
        'ipywidgets',
        'widgetsnbextension',
        'pandas-profiling',
        'networkx',
        'pandas',
        
    ],
    zip_safe=False,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vaultree/venumML",
    license="BSD-3-Clause",
    classifiers=[
        "License :: OSI Approved :: BSD License",
    ],
)
