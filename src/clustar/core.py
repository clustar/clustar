import numpy as np
import astropy.io
from clustar.clustarray import ClustArray
from clustar import group, graph, fit


class ClustarData(object):

    class Image(object):

        def __init__(self, data):
            self.data = data
            self.clean = None
            self.x = None
            self.y = None
            self.pos = None
            self.nonzero = None
            self.setup()

        def setup(self):
            x = range(self.data.shape[1])
            y = range(self.data.shape[0])
            self.x, self.y = np.meshgrid(x, y)
            self.pos = np.dstack((self.x, self.y))
            self.data = np.nan_to_num(self.data, nan=0)

    class Group(object):

        class _Image(object):

            def __init__(self, bounds):
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

            def __init__(self):
                self.data = None
                self.clean = None
                self.pos = None
                self.inside = None
                self.outside = None
                self.output = None

        class _Fit(object):

            def __init__(self):
                self.rv = None
                self.bvg = None
                self.ellipse = None
                self.major_peaks = None
                self.minor_peaks = None

        class _Stats(object):

            def __init__(self):
                self.x_bar = None
                self.y_bar = None
                self.x_var = None
                self.y_var = None
                self.x_len = None
                self.y_len = None
                self.covariance = None
                self.covariance_matrix = None
                self.rho = None
                self.eigen_values = None
                self.eigen_vectors = None
                self.radians = None
                self.degrees = None

        class _Metrics(object):

            def __init__(self):
                self.standard_deviation = None
                self.variance = None
                self.average = None
                self.weighted_average = None

        def __init__(self, bounds):
            self.image = self._Image(bounds)
            self.res = self._Residuals()
            self.fit = self._Fit()
            self.stats = self._Stats()
            self.metrics = self._Metrics()
            self.flag = False

    class Params(object):

        def __init__(self, args):
            self.alpha = 0.2
            self.buffer_size = 10
            self.group_size = 50
            self.group_factor = 0
            self.split_binary = False
            self.subgroup_factor = 0.5
            self.metric = "variance"
            self.threshold = 0.01
            self.smoothing = 5
            self.clip = 0.75
            self.extract(args)

        def extract(self, args):
            for key in args:
                if key not in vars(self).keys():
                    raise KeyError(f"Invalid keyword '{key}' has been " +
                                   "passed into the ClustarData object.")
                setattr(self, key, args[key])

    def __init__(self, path, **kwargs):
        self.image = None
        self.header = None
        self.params = self.Params(kwargs)
        self.groups = []
        self.load_file(path)
        self.flag = False
        self.denoise()
        self.setup()

    def load_file(self, path):
        file = astropy.io.fits.open(path)
        data = file[0].data[0, 0, :, :]
        self.image = self.Image(data)
        self.header = file[0].header

    def setup(self):
        self = group.arrange(self)
        self.build()
        if self.params.split_binary:
            self = group.detect(self)
            self.build()
        self.evaluate()

    def build(self):
        self = group.rectify(self)
        self = group.merge(self)
        self = group.refine(self)
        self = group.extract(self)
        self = group.calculate(self)

    def evaluate(self):
        self = fit.compute_fit(self)
        self = fit.compute_ellipse(self)
        self = fit.compute_metrics(self)
        self = fit.compute_peaks(self)
        self = fit.validate(self)

    def update(self, **kwargs):
        self.params.extract(kwargs)
        self.setup()

    def reset(self, **kwargs):
        self.params = self.Params(kwargs)
        self.setup()

    def denoise(self):
        ca = ClustArray(self.image.data)
        ca.denoise()
        image = ca.denoised_arr
        image[image < ca.noise_est * 5] = 0
        image = np.nan_to_num(image, nan=0)
        #         image = self.image.data.copy()
        #         std = np.std(image)
        #         image[image < std * 5] = 0
        self.image.nonzero = np.dstack(np.nonzero(image))[0]
        self.image.clean = image

    def identify(self, vmin=None, vmax=None, show=True, dpi=180):
        graph.identify_groups(self, vmin, vmax, show, dpi)
