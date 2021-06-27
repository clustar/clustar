"""
Contains the 'ClustarData' class, which is responsible for executing the
entire project pipeline for detecting groups in a single FITS image; this 
class also collects and stores all relevant data, statistics, and variables in
this pipeline.

Visit <https://clustar.github.io/> for additional information.
"""


from clustar import denoise, group, graph, fit
import astropy.io.fits
import numpy as np


class ClustarData(object):
    """
    A class for executing the entire pipline for detecting groups in a FITS 
    image and for storing all relevant data associated with each group.
    
    Attributes
    ----------
    path : str
        Path to FITS file.
    
    image : Image
        Internal class for storing FITS image variables.
    
    params : Params
        Internal class for specifying the ClustarData parameters.
    
    groups : list
        List of 'Group' objects extracted from the given FITS image.
    
    flag : bool
        True if any detected group in the FITS image is flagged for manual 
        review, otherwise false.
    
    Methods
    -------
    update(**kwargs)
        Updates 'Params' object with the specified arguments and executes the
        entire pipeline.
    
    reset(**kwargs)
        Resets 'Params' object to the default values, then updates 'Params' 
        object with the specified arguments and executes the entire pipeline.
    
    identify(vmin=None, vmax=None, show=True, dpi=180)
        Displays the FITS image and identifies the groups in green, orange, 
        or red rectangles, which are defined as:
        1. 'Green' denotes that the group is not flagged for manual review
        2. 'Orange' denotes that the group is not flagged for manual review, 
           but the group is smaller than the beam size.
        3. 'Red' denotes that the group is flagged for manual review.
        Beam size is the white oval shown on the bottom right corner of 
        the FITS image.
    
    Examples
    --------
    Create the 'ClustarData' object by specifying the path to FITS file. 
    >>> cd = ClustarData(path='~/data/example.fits', threshold=0.025)
    
    Visualize the detected groups.
    >>> cd.identify()
    
    Access individual 'Group' objects.
    >>> cd.groups
    
    Notes
    -----
    Visit <https://clustar.github.io/> for additional information.
    """

    class Image(object):
        """
        An internal class for storing FITS image variables.
        
        Attributes
        ----------
        clean : ndarray
            Data from the FITS image after denoising process.
        
        x : ndarray
            Index values of the 'x' position from the data.
            
        y : ndarray
            Index values of the 'y' position from the data.
        
        pos : ndarray
            Index values of the data, given as (x, y).
            
        nonzero : ndarray
            Index values of nonzero points in the data.
        
        std : ndarray
            Standard deviation values from each block in the grid composed
            in the denoise process; used to calculate the noise statistic.
        
        rms : ndarray
            Root mean squared values from each block in the grid composed
            in the denoise process; used to calculate the noise statistic.
        
        noise : float
            Noise statistic generated for the denoise process; values less 
            than "noise" times "sigma" are set to zero.
        
        major : float
            Length of the major axis for the beam.
        
        minor : float
            Length of the minor axis for the beam.
        
        degrees : float
            Degrees of rotation for the beam.
        
        area : float
            Number of points inside the beam; used to identify groups smaller
            than the beam size.
        """

        def __init__(self, data, header):
            """
            Parameters
            ----------
            data : ndarray
                Raw data from the FITS image; must be 2-D.

            header : dict
                Header dictionary stored in FITS file.
            
            Raises
            ------
            KeyError
                If the following keys are missing from the FITS header: 
                'BMAJ', 'BMIN', 'BPA', 'CDELT1', 'CDELT2', and 'OBJECT'.
            """
            self.data = data
            self.header = header
            self._setup()

        def _setup(self):
            x = range(self.data.shape[1])
            y = range(self.data.shape[0])
            self.x, self.y = np.meshgrid(x, y)
            self.pos = np.dstack((self.x, self.y))
            header = dict(self.header)
            keys = ['BMAJ', 'BMIN', 'BPA', 'CDELT1', 'CDELT2', 'OBJECT']
            for key in keys:
                if key not in header.keys():
                    raise KeyError("FITS header is missing the " +
                                   f"keyword '{key}'; double check " +
                                   "the file type specification.")
            # Specify beam parameters.
            self.major = header['BMAJ']/abs(header['CDELT1'])
            self.minor = header['BMIN']/abs(header['CDELT2'])
            self.degrees = header['BPA']
            self.area = np.pi * self.major/2 * self.minor/2

    class Group(object):
        """
        An internal class for storing variables associated to a detection.
        
        Attributes
        ----------
        image : _Image
            Internal subclass for storing image variables.
        
        res : _Res
            Internal subclass for storing residual variables.
        
        fit : _Fit
            Internal subclass for storing fit variables.
        
        stats : _Stats
            Internal subclass for storing statistics.
        
        metrics : _Metrics
            Internal subclass for storing the evaluated metrics.
        
        flag : bool
            Determines whether this group is marked for manual review.
        """

        class _Image(object):
            """
            An internal subclass for storing image variables associated to a
            detection.
            
            Attributes
            ----------
            data : ndarray
                Subset of raw data from the FITS image identifying the group.
            
            clean : ndarray
                Data of the group after the denoising process.
            
            x : ndarray
                Index values of the 'x' position from the group data.
            
            y : ndarray
                Index values of the 'y' position from the group data.
            
            pos : ndarray
                Index values of the group data, given as (x, y).
            
            nonzero : ndarray
                Index values of nonzero points in the group data.
            
            ref : list
                List containing the minimum row value and minimum column value
                of the group data.
            
            limit : list
                List containing the maximum row value and maximum column value
                of the overall FITS image.
            """

            def __init__(self, bounds):
                """
                Parameters
                ----------
                bounds : list
                    List of four integers corresponding to minimum row value, 
                    maximum row value, minimum column value, and maximum column
                    value in this order.
                """
                self.bounds = bounds
                self.data = None
                self.clean = None
                self.x = None
                self.y = None
                self.ref = None
                self.limit = None
                self.pos = None
                self.nonzero = None

        class _Residuals(object):
            """
            An internal subclass for storing residual variables associated to
            a detection.
            
            Attributes
            ----------
            data : ndarray
                Residuals computed in the fitting process. Precisely, they
                are [1 - ("bivariate Gaussian model" / "group data")].
            
            clean : ndarray
                Residuals computed in the fitting process, where points 
                outside of the ellipse are set to zero.
            
            pos : ndarray
                Index values of the residual data, given as (x, y).
            
            inside : ndarray
                Subset of index values that lie inside of the ellipse.
            
            outside : ndarray
                Subset of index values that lie outside of the ellipse.
            
            output : array_like
                List of residuals that lie inside of the ellipse; the result 
                of the evaluation metric that is computed on this list is 
                compared to the specified threshold; this determines which
                groups are flagged for manual review.
            """

            def __init__(self):
                self.data = None
                self.clean = None
                self.pos = None
                self.inside = None
                self.outside = None
                self.output = None

        class _Fit(object):
            """
            An internal subclass for storing fit variables associated to a 
            detection.
            
            Attributes
            ----------
            rv : multivariate_normal_frozen
                Frozen multivariable normal distribution generated from the 
                group statistics.
            
            bvg : ndarray
                Results of the multivariate normal probability density 
                function evaluated at the points specified by the group data.
            
            ellipse : Polygon
                Polygon object containing the points that generate an ellipse
                corresponding to the multivariate normal distribution.
            
            major_peaks : int
                Number of local maximas along the major axis of the ellipse.
            
            minor_peaks : int
                Number of local maximas along the minor axis of the ellipse.
            """

            def __init__(self):
                self.rv = None
                self.bvg = None
                self.ellipse = None
                self.major_peaks = None
                self.minor_peaks = None

        class _Stats(object):
            """
            An internal subclass for storing statistics associated to a 
            detection.
            
            Attributes
            ----------
            x_bar : float
                Average of index values in the 'x' position weighted by the
                corresponding group data.
                
            y_bar : float
                Average of index values in the 'y' position weighted by the 
                corresponding group data.
            
            x_var : float
                Variance of index values in the 'x' position weighted by the 
                corresponding group data.
            
            y_var : float
                Variance of index values in the 'y' position weighted by the 
                corresponding group data.
            
            covariance : float
                Covariance of the index values weighted by the corresponding
                group data.                
            
            covariance_matrix : array_like
                Covariance matrix for the multivariate normal that is used in
                the fitting process.
                
            rho : float
                Correlation coefficient computed from the covariance matrix.
            
            eigen_values : array_like
                Eigenvalues obtained from the eigendecomposition of the
                covariance matrix.
            
            eigen_vectors : array_like
                Eigenvectors obtained from the eigendecomposition of the
                covariance matrix.
            
            x_len : float
                Length of the major axis of the ellipse in pixels.
            
            y_len : float
                Length of the minor axis of the ellipse in pixels.
            
            radians : float
                Rotation of ellipse denoted in radians.
            
            degrees : float
                Rotation of ellipse denoted in degrees.
            """

            def __init__(self):
                self.x_bar = None
                self.y_bar = None
                self.x_var = None
                self.y_var = None
                self.covariance = None
                self.covariance_matrix = None
                self.rho = None
                self.eigen_values = None
                self.eigen_vectors = None
                self.x_len = None
                self.y_len = None
                self.radians = None
                self.degrees = None

        class _Metrics(object):
            """
            An internal subclass for storing the evaluated metrics associated
            to a detection.
            
            Attributes
            ----------
            standard_deviation : float
                Standard deviation of the output residuals for the group.
                
            variance : float
                Variance of the output residuals for the group.
            
            average : float
                Mean of the output residuals for the group.
            
            weighted_average : float
                Mean of the output residuals weighted by the group data.
            """
            
            def __init__(self):
                self.standard_deviation = None
                self.variance = None
                self.average = None
                self.weighted_average = None

        def __init__(self, bounds):
            """
            Parameters
            ----------
            bounds : list
                List of four integers corresponding to minimum row value, 
                maximum row value, minimum column value, and maximum column
                value in this order.
            """
            self.image = self._Image(bounds)
            self.res = self._Residuals()
            self.fit = self._Fit()
            self.stats = self._Stats()
            self.metrics = self._Metrics()
            self.flag = False

    class Params(object):
        """
        An internal class for specifying the ClustarData parameters.
        
        Attributes
        ----------
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
            following: "standard_deviation", "variance", "average", or 
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
        """

        def __init__(self, args):
            """
            Parameters
            ----------
            args : dict
                Dictionary of keyword arguments; see 'Attributes' for keys.
                
            Raises
            ------
            KeyError
                If specified key in 'args' does not match the label of the 
                specified attributes.
            """
            self.radius_factor = 1
            self.chunks = 3
            self.quantile = 0.5
            self.apply_gradient = True
            self.sigma = 5
            self.alpha = 0.2
            self.buffer_size = 10
            self.group_size = 50
            self.group_factor = 0
            self.split_binary = False
            self.subgroup_factor = 0.5
            self.metric = "variance"
            self.threshold = 0.01
            self.evaluate_peaks = False
            self.smoothing = 5
            self.clip = 0.75
            self._extract(args)

        def _extract(self, args):
            for key in args:
                if key not in vars(self).keys():
                    raise KeyError(f"Invalid keyword '{key}' has been " +
                                   "passed into the ClustarData object.")
                setattr(self, key, args[key])

    def __init__(self, path, **kwargs):
        """
        Parameters
        ----------
        path : str
            Path to FITS file.
        
        **kwargs : optional
            See '~clustar.core.ClustarData.params' for other possible 
            arguments.
        """
        self.path = path
        self.params = self.Params(kwargs)
        self.groups = []
        self.flag = False
        self._load_file()
        self._setup()

    def _load_file(self):
        file = astropy.io.fits.open(self.path)
        data = file[0].data[0, 0, :, :]
        header = file[0].header
        self.image = self.Image(data, header)

    def _setup(self):
        self = denoise.resolve(self)
        self = group.arrange(self)
        self._build()
        if self.params.split_binary:
            self = group.detect(self)
            self._build()
        self._evaluate()

    def _build(self):
        self = group.rectify(self)
        self = group.merge(self)
        self = group.refine(self)
        self = group.extract(self)
        self = group.screen(self)
        self = group.calculate(self)

    def _evaluate(self):
        self = fit.compute_fit(self)
        self = fit.compute_ellipse(self)
        self = fit.compute_metrics(self)
        self = fit.compute_peaks(self)
        self = fit.validate(self)

    def update(self, **kwargs):
        """
        Updates 'Params' object with the specified arguments and executes the
        entire pipeline.
        
        Parameters
        ----------       
        **kwargs : optional
            See '~clustar.core.ClustarData.params' for other possible 
            arguments.
        """
        self.params.extract(kwargs)
        self._setup()

    def reset(self, **kwargs):
        """
        Resets 'Params' object to the default values, then updates 'Params' 
        object with the specified arguments and executes the entire pipeline.
        
        Parameters
        ----------       
        **kwargs : optional
            See '~clustar.core.ClustarData.params' for other possible 
            arguments.
        """
        self.params = self.Params(kwargs)
        self._setup()

    def identify(self, vmin=None, vmax=None, show=True, dpi=180):
        """
        Displays the FITS image and identifies the groups in green, orange, or
        red rectangles, which are defined as:
        1. Green denotes that the group is not flagged for manual review
        2. Orange denotes that the group is not flagged for manual review, but
           the group is smaller than the beam size.
        3. Red denotes that the group is flagged for manual review.
        Beam size is the white oval shown on the bottom right corner of 
        the FITS image.
        
        Parameters
        ----------
        vmin : float, optional
            Lower bound for the shown intensities. 
        
        vmax : float, optional
            Upper bound for the shown intensities.
            
        show : bool, optional
            Determines whether the groups should be identified. If false, the
            rectangles identifying the groups are not drawn.
        
        dpi : int, optional
            Dots per inch.
        """
        graph.identify_groups(self, vmin, vmax, show, dpi)
