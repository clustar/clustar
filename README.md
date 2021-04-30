
## Clustar

Release: 1.1.10

Date: April 23, 2021

Clustar is a python package using machine learning algorithms for processing and analyzing prostars/protoplanetary disks. 

Links to our [paper]()

## Import

-- how to install clustar -- 

```python
$ pip install clustar
```

Imports and basic examples can be found on the [Clustar github page](https://clustar.github.io/)

## Motivation

The motivation for using clustar is to identify prostars/protoplanetary disks stored in FITS files. These files contain grayscale images represented as 2d arrays, with each pixals containing the intensity values, and the header information about the telescope observational parameters. Clustar simplifies and expediates the identification pipeline of FITS files by automating the preprocessing, grouping, and fitting for a large amount of FITS files. Clustar is optimized in its codebase primarily in its efficacy in identifying non-bivariate Gaussian like substructures and is trained and tested upon the dataset curated by [Tobin et al.](https://ui.adsabs.harvard.edu/abs/2020ApJ...890..130T/abstract).

### Preprocessing

Clustar crops the input image from a square to a circle while retaining the original dimension. The pixals not in the croped circle will be masked. This is done to alleviate the higher noise around the edges of the image introduced by the telescope. After cropping, clustar utilizes a technique known as sigma clipping to filter out the pixals that are less than 5 times the RMS (Root Mean Square) statistic. 

### Grouping

### Fitting

### Summary

Clustar should be utilized to identify protostars and other potential celestial objects that are suspected to be non-bivariate Gaussian. The t-SNE clustering methods further identify images with substructures that may be of interest. Anyone that works with FITSits files can utilize the different methods idependently or as a pipeline to preprocess, group, fit, and cluster their data.


```python

```
