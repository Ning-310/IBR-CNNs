# IBR-CNNs
Imaging-based representation (IBR) of thermal proteomes.

For temperature-series thermal proteomes, proteins were quantified by high-resolution mass spectrometry to obtain protein denaturation curves, which can reflect thermal stability and the change of ligand-induced thermal stability of proteins. Thus, protein denaturation curves might be directly used as an imaging feature, for the identification of potential drug targets. 

## Requirements

The main requirements are listed below:

* Python 3.5
* Numpy
* Scikit-Learn
* matplotlib
* Keras
* Pandas


## The description of iBM source

* processing.py

    The code is used to thermal proteomics data preprocessing.

* IBR.py

    The code is used to imaging-based representation of thermal proteomes.

* CNNs.py

    The code is convolutional neural networks framework for model training and evaluation.

* evaluate.py

    The code is used to model evaluation.

* ROC.py

    The code is used to illustrate the receiver operating characteristic (ROC) curve based on sensitivity and 1-specificity scores, and compute the AUC value.

* IBR-CNNs.model 

    The model of IBR-CNNs.
