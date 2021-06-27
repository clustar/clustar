"""
Internal module for testing the 'ClustarData' object. 

Visit <https://clustar.github.io/> for additional information.
"""


from clustar.core import ClustarData
from scipy import stats
import numpy as np

class ClustarBase(ClustarData):
    """
    A class that inherits from 'ClustarData'; however, the FITS image data is
    user generated for testing purposes.  
    
    Attributes
    ----------
    params : Params
        Internal class for specifying the ClustarData parameters.
    
    groups : list
        List of 'Group' objects extracted from the given FITS image.
    
    flag : bool
        True if any detected group in the FITS image is flagged for manual
        review, otherwise false.
    """
    class Image(object):

        def __init__(self, data, **kwargs):
            """
            Parameters
            ----------
            data : ndarray
                Raw data from the FITS image; must be 2-D.
            
            major : float, optional
                Length of the major axis for the beam.

            minor : float, optional
                Length of the minor axis for the beam.
        
            degrees : float, optional
                Degrees of rotation for the beam.
            """
            self.data = data
            self.major = 10
            self.minor = 10
            self.degrees = 25
            self._setup(kwargs)

        def _setup(self, args):
            x = range(self.data.shape[1])
            y = range(self.data.shape[0])
            self.x, self.y = np.meshgrid(x, y)
            self.pos = np.dstack((self.x, self.y))
            for key in args:
                if key not in vars(self).keys():
                    raise KeyError(f"Invalid keyword '{key}' has been " +
                                   "passed into the ClustarDataTester object.")
                setattr(self, key, args[key]) 
            self.area = np.pi * self.major/2 * self.minor/2
    
    def __init__(self, **kwargs):
        self.params = self.Params(kwargs)
        self.groups = []
        self.flag = False

    def generate_data(self, x_bar, y_bar, x_var, y_var, covariance, sigma,
                      n_rows, n_cols):
        """
        Generates a singular bivariate gaussian, given the specified 
        parameters.
        
        Parameters
        ----------
        x_bar : float
            Average of index values in the 'x' position.
                
        y_bar : float
            Average of index values in the 'y' position.
        
        x_var : float
            Variance of index values in the 'x' position.
        
        y_var : float
            Variance of index values in the 'y' position.
        
        covariance : float
            Covariance of the index values.
        
        sigma : float
            Factor multiplied to noise level to determine the cutoff point,
            where values less than this threshold are set to zero.

        n_rows, n_cols : int
            Dimensions of the computed data.

        Returns
        -------
        ndarray
        """
        x = np.arange(0, n_rows, 1)
        y = np.arange(0, n_cols, 1)
        x, y = np.meshgrid(x, y)
        pos = np.dstack((x, y))

        cov_mat = [[x_var, covariance], [covariance, y_var]]
        rv = stats.multivariate_normal([x_bar, y_bar], cov_mat)
        data = rv.pdf(pos)

        std = np.std(data)
        data[data < std * sigma] = 0
        return data

    def generate_image(self, params, n_rows, n_cols, **kwargs):
        """
        Generates multiple bivariate gaussians, given a list of the specified
        parameters.

        Parameters
        ----------
        params : list
            List of 'generate_data()' parameters in the following order: 
            'x_bar', 'y_bar', 'x_var', 'y_var', 'covariance', and 'sigma'. 

        n_rows, n_cols : int
            Dimensions of the computed data.

        Returns
        -------
        ndarray
        """
        data = np.zeros((n_rows, n_cols))
        for param in params:
            x_bar, y_bar, x_var, y_var, cov, sigma = param
            data += self.generate_data(x_bar, y_bar, x_var, y_var,
                                       cov, sigma, n_rows, n_cols)
        self.image = self.Image(data, **kwargs)
