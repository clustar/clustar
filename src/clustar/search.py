"""
Contains the 'Clustar' hierarchical class, which is responsible for 
transforming all available FITS images in a specified directory into their 
respective 'ClustarData' objects. 'Clustar' class uses multithreading to
expedite the processing of the FITS images.

Visit <https://clustar.github.io/> for additional information.
"""


from clustar.core import ClustarData
from tqdm import tqdm
import concurrent
import numpy as np
import os
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")


class Clustar(object):
    """
    A class for executing the entire pipline for detecting groups from a 
    directory of FITS files and for storing all relevant data associated 
    with each file.
    
    Parameters
    ----------
    **kwargs : optional
        Same as '~clustar.core.ClustarData.params' arguments. Copy of that
        documentation is listed below for convenience.
    
    Keywords
    --------
    radius_factor : float
        Factor mulitplied to radius to determine cropping circle in the 
        denoising process; must be within the range [0, 1].

    chunks : int
        Number of chunks to use in a grid; must be an odd number.

    quantile : float
        Quantile of RMS to determine the noise level; must be within the 
        range [0, 1].

    apply_gradient : bool
        Determine if the FITS image should be multiplied by a gradient
        in order to elevate central points; similar to multiplying the
        FITS image by the associated 'pb' data.

    sigma : float
        Factor multiplied to noise level to determine the cutoff point,
        where values less than this threshold are set to zero.

    alpha : float
        Determines the size of the ellipse in relation to the chi-squared 
        distribution; must be within the range (0, 1).

    buffer_size : int
        Number of points considered outside of the group range. For in-
        stance, given a 1-d group range of [10, 20], the algorithm checks
        for nonzero points within the range [5, 25] when the 'buffer_size'
        is 5.

    group_size : int
        Minimum number of nonzero points that determines a group.

    group_factor : float
        Ratio between [0, 1] that specifies the minimum number of 
        nonzero points that determines a group in relation to the number
        of nonzero points in the largest group.

    metric : str
        Method used for evaluating the groups; must be one of the 
        following: "standard_deviation", "variance", "average", and 
        "weighted_average".

    threshold : float
        Cutoff point that determines which groups are flagged for manual
        review, given the specified metric. 

    split_binary : bool
        Experimental; determine whether binary subgroups identified 
        within a group should be split into individual groups.

    subgroup_factor : float
        Experimental; ratio between [0, 1] that specifies the subgroup 
        range in terms of the absolute maximum intensity.

    evaluate_peaks : bool
        Experimental; determine whether the peaks of the output residuals
        should be taken into consideration in the flagging process.

    smoothing : int
        Experimental; size of window used in the moving average smoothing
        process for peak evaluation.

    clip : float
        Experimental; determines the percentage of tail values that are 
        trimmed for peak evaluation.
    
    Attributes
    ----------
    data : list
        List of 'ClustarData' objects corresponding to the FITS file in the 
        directory.
        
    errors : list
        List of FITS files that raised an error in the processing.
    
    Methods
    -------
    run(directory)
        Main method for executing the pipeline for group detection.
    
    display(category='summary')
        Generates a Pandas data frame object containing the specified
        variables.    
    
    Examples
    --------
    Setup 'Clustar' object.
    >>> cs = Clustar(radius_factor=0.95, threshold=0.025)
    
    Execute pipeline on directory containing FITS files.
    >>> cs.run(directory='~/data/')
    
    Access individual 'ClustarData' objects.
    >>> cs.data
    
    Check which FITS files raised an error.
    >>> cs.errors
    
    Inspect 'ClustarData' variables for all groups in each FITS file.
    >>> cs.display(category='all')
    
    Notes
    -----
    Visit <https://clustar.github.io/> for additional information.
    """
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.data = []
        self.errors = []
        
    def _execute(self, path):
        try:
            cd = ClustarData(path, **self.params)        
            return cd
        except:
            return os.path.basename(path)
    
    def _extract(self, cd, index):
        observations = []
        image_info = {'index': index,
                      'file': os.path.basename(cd.path),
                      'object': cd.image.header['OBJECT'],
                      'noise': cd.image.noise,
                      'beam_size': cd.image.area}
        if len(cd.groups) == 0:
            return [image_info]
        for group in cd.groups:
            row_min, row_max, col_min, col_max = group.image.bounds
            nonzero = len(group.image.nonzero)
            major_peaks = group.fit.major_peaks
            minor_peaks = group.fit.minor_peaks
            group_info = {'row_min': row_min,
                          'row_max': row_max,
                          'col_min': col_min,
                          'col_max': col_max,
                          'group_size': nonzero,
                          'major_peaks': major_peaks,
                          'minor_peaks': minor_peaks,
                          'flag': group.flag}
            group_stats = vars(group.stats).copy()
            group_metrics = vars(group.metrics).copy()
            info = dict(image_info, **group_info, 
                        **group_stats, **group_metrics)
            observations.append(info)
        return observations
    
    def run(self, directory):
        """
        Main method for executing the pipeline for group detection.
        
        Parameters
        ----------
        directory : str
            Path to folder containing FITS files.
        
        Raises
        ------
        AssertionError
            If there are no FITS files at the specified directory. 
        """
        files = os.listdir(directory)
        files = [file for file in files if file.endswith('.fits')]
        assert len(files) > 0, "Could not find any FITS files " + \
                               "at specified directory."
        paths = [os.path.join(directory, file) for file in files]
        counter = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(self._execute, paths)
            progress = tqdm(results, total=len(paths), 
                            desc='Initializing',
                            unit='file', colour='green')
            for cd in progress:
                if type(cd) is str:
                    self.errors.append(cd)
                else:
                    file_name = cd.image.header['OBJECT']
                    if cd.flag:
                        counter += 1
                    self.data.append(cd)
                progress.set_description(f"File '{file_name}'")
                progress.set_postfix(num_errors=len(self.errors),
                                     num_flags=counter)
    
    def display(self, category='summary'):
        """
        Generates a Pandas data frame object containing the specified
        variables.
        
        Parameters
        ----------
        category : str, optional
            If category is 'summary', then a subset of 'ClustarData' variables
            are returned in the data frame. If category is 'all', then all 
            singular 'ClustarData' variables are returned in the data frame.
        
        Returns
        -------
        DataFrame
        """
        if len(self.data) == 0:
            return pd.DataFrame()
        
        table = []       
        for index, cd in enumerate(self.data):
            table += self._extract(cd, index)
        
        categories = ['all', 'summary']
        category = category.lower()
        if category not in categories:
            raise KeyError("Could not display specified category; " + 
                           f"valid categories are {categories}.")
        
        metric = self.data[-1].params.metric
        df = pd.DataFrame(table)
        if category == 'all':
            return df
        elif category == 'summary':
            return df[['index', 'object', 'flag', 'noise', 'beam_size',
                       'group_size', metric, 'major_peaks', 'minor_peaks']]
