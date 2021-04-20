# Example PyPI (Python Package Index) Package
import unittest
import numpy as np
import pandas as pd
from scipy import stats
import pkg_resources
# import matplotlib
# from matplotlib import pyplot as plt
# from matplotlib.patches import Ellipse
# from astropy.table import Table
# import astropy.io
# from astroML.stats import fit_bivariate_normal
# from astroML.stats.random import bivariate_normal
from clustar import *
from clustar import clustarray

class parameter_fitting(object):
    def load_example(file_name):
        stream = pkg_resources.resource_stream(__name__, file_name)
        stream =  pd.read_csv(stream)
        return stream.to_numpy()

    def bivariate_gaussian_fit(array):
        x = range(array.shape[1])
        y = range(array.shape[0])
        x, y = np.meshgrid(x, y)
        pos = np.dstack((x, y))
        image_data = np.nan_to_num(array, nan=0)
        x_bar = np.average(x,weights = image_data)
        y_bar = np.average(y, weights=image_data)
        x_var = np.average((x - x_bar) ** 2, weights=image_data)
        y_var = np.average((y - y_bar) ** 2, weights=image_data)
        cov = np.average(x * y, weights=image_data) - x_bar * y_bar
        cov_mat = np.array([[x_var, cov], [cov, y_var]])
        rv = stats.multivariate_normal([x_bar, y_bar], cov_mat)
        bvg = rv.pdf(pos)
        return [x_bar, y_bar, x_var, y_var, cov_mat, rv, bvg] #Returns the parameters in a list

    # Testing Portion:
    example_array = load_example('data/example.csv')

    #Fitting:
    parameters = bivariate_gaussian_fit(example_array)
    dataframe = pd.DataFrame(parameters,index=['xbar','ybar', 'xvar', 'yvar' , 'cov_mat','rv','bvg'])
    print(dataframe)

    #testing denoising
    image = load_example('data/test_fits.csv')
    pb = load_example('data/test_pb.csv')

    clust = clustarray.ClustArray(image)
    clust.denoise(pb_array = pb)
    print('Noise level: ' + str(clust.noise_est))
