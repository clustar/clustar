class ClustarData(object):
    class Image(object):

        def __init__(self, data):
            self.data = data
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
            self.nonzero = np.dstack(np.nonzero(self.data))[0]

    class Group(object):
        class _Image(object):

            def __init__(self):
                self.data = None
                self.bounds = None
                self.x = None
                self.y = None
                self.ref = None
                self.limit = None
                self.pos = None

        class _Residuals(object):

            def __init__(self):
                self.pos = None
                self.inside = None
                self.outside = None

        class _Fit(object):

            def __init__(self):
                self.bvg = None
                self.ellipse = None

        class _Stats(object):

            def __init__(self):
                self.x_bar = None
                self.y_bar = None
                self.x_var = None
                self.y_var = None
                self.cov = None
                self.cov_mat = None
                self.rho = None
                self.eigen_values = None
                self.eigen_vectors = None

        class _Metrics(object):

            def __init__(self):
                self.standard_deviation = None
                self.variance = None
                self.average = None
                self.weighted_average = None

        def __init__(self):
            self.image = self._Image()
            self.residuals = self._Residuals()
            self.fit = self._Fit()
            self.stats = self._Stats()
            self.metrics = self._Metrics()

    def __init__(self, data):
        self.image = self.Image(data)
        self.groups = []


if __name__ == "__main__":
    import numpy as np

    image = np.zeros((10, 10))
    image[1:4, 1:4] = 1
    image[1:4, 6:9] = 1
    image[6:9, 1:4] = 1
    image[6:9, 6:9] = 1
    cd = ClustarData(image)
    print(cd.image.data)
