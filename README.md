
### Clustar

Release: 1.1.10

Date: April 23, 2021

### Import

-- how to install clustar -- 

''' pip install clustar '''

### Motivation

The motivation for using clustar is to identify prostars/ protoplanetary disks found from FITS files. These files are represented as 2d array's containing intensities at each point along with the header information about the telescope observational parameters. Clustar simplifies and expediates the identification pipeline of FITS files by automating the preprocessing, grouping, and fitting for a group of FITS files. Clustar is optimized in its codebase primarily in its efficacy in identifying non-bivariate Gaussian like substructures and is tested upon the Tobin dataset.

### Preprocessing

Clustar crops the input image from a square dimension to a circle. This is done to alleviate the higher noise around the edges of the image. After cropping, clustar utilizes a technique known as sigma clipping to filter out the data points that are 5 times the RMS statistic. 

### Grouping

### Fitting

### Summary

Clustar should be utilized to identify protostars and other potential celestial objects that are suspected to be non-bivariate Gaussian. The t-SNE clustering methods further identify images with substructures that may be of interest. Anyone that works with FITSits files can utilize the different methods idependently or as a pipeline to preprocess, group, fit, and cluster their data.


```python

```
