# clustar
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Clustar\n",
    "\n",
    "Release: 1.1.10\n",
    "\n",
    "Date: April 23, 2021"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Import"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "-- how to install clustar -- "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "''' pip install clustar '''"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Motivation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The motivation for using clustar is to identify prostars/ protoplanetary disks found from fits files. These files are represented as 2d array's containing intensities at each point along with the header information about the telescope observational parameters. Clustar simplifies and expediates the identification pipeline of fits files by automating the preprocessing, grouping, fitting, and non supervised learning for a group of fits files. Clustar is optimized in its codebase primarily in its efficacy in identifying non - bivariate Gaussian like substructures and is tested on the Tobin dataset."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Preprocessing"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Clustar crops the input image from a square dimension to a circle. This is done to alleviate the higher noise around the edges of the image. After cropping, clustar utilizes a technique known as sigma clipping to filter out the data points that are 5 times the RMS statistic. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Grouping"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Fitting"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Summary"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Clustar should be utilized to identify protostars and other potential celestial objects that are suspected to be non-bivariate Gaussian. The t-SNE clustering methods further identify images with substructures that may be of interest. Anyone that works with fits files can utilize the different methods idependently or as a pipeline to preprocess, group, fit, and cluster their data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
