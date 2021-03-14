
class ClustarData(object):

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


if __name__ == "__main__":
    import numpy as np
    image = np.zeros((10, 10))
    image[1:4, 1:4] = 1
    image[1:4, 6:9] = 1
    image[6:9, 1:4] = 1
    image[6:9, 6:9] = 1
    cd = ClustarData()
    cd.image.data = image
    print(cd.image.data)
