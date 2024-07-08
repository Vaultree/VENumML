from setuptools import setup, find_packages

setup(
    name='venumML',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        scipy
        deepface
        tensorflow
        matplotlib
        numpy
        scikit-learn
        transformers
        tqdm
        torch
        ipywidgets 
        widgetsnbextension 
        pandas-profiling
    ],
)
