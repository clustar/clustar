# Example PyPI (Python Package Index) Package
import unittest
from astropy.io import fits
import astropy.io
import numpy as np
import matplotlib
from astropy.table import Table
import astropy.io
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from astroML.stats import fit_bivariate_normal
from astroML.stats.random import bivariate_normal
import numpy as np
import pandas as pd
import pkg_resources
class parameter_fitting(object):
    def load_example():
        stream = pkg_resources.resource_stream(__name__, 'data/example.csv')
        stream =  pd.read_csv(stream)
        return stream.to_numpy()

    def bivariate_gaussian_fit(np_array):
        print(np_array)
    
    # Testing Portion:
    example_array = load_example()
    
    #Fitting:
    bivariate_gaussian_fit(example_array)
    

