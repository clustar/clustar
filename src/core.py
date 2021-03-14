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

            def __init__(self, bounds):
                self.data = None
                self.bounds = bounds
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
                self.covariance = None
                self.covariance_matrix = None
                self.rho = None
                self.eigen_values = None
                self.eigen_vectors = None

        class _Metrics(object):

            def __init__(self):
                self.standard_deviation = None
                self.variance = None
                self.average = None
                self.weighted_average = None

        def __init__(self, bounds):
            self.image = self._Image(bounds)
            self.residuals = self._Residuals()
            self.fit = self._Fit()
            self.stats = self._Stats()
            self.metrics = self._Metrics()

    def __init__(self, data):
        self.image = self.Image(data)
        self.groups = []

    def setup(self):
        self.arrange_groups()
        self.compute_stats()

    def arrange_groups(self):
        for group in self.groups:
            row_min, row_max, col_min, col_max = group.image.bounds
            x = np.arange(col_min, col_max + 1, 1)
            y = np.arange(row_min, row_max + 1, 1)
            group.image.x, group.image.y = np.meshgrid(x, y)
            group.image.pos = np.dstack((group.image.y, group.image.x))

            n_rows = group.image.pos.shape[0]
            n_cols = group.image.pos.shape[1]

            group.image.data = np.zeros((n_rows, n_cols))
            for r in range(n_rows):
                for c in range(n_cols):
                    group.image.data[r, c] = \
                        self.image.data[tuple(group.image.pos[r, c])]

            group.image.ref = [col_min, row_min]
            group.image.limit = [self.image.data.shape[0],
                                 self.image.data.shape[1]]

    def compute_stats(self):
        for group in self.groups:
            stats_ = group.stats
            image_ = group.image
            try:
                stats_.x_bar = np.average(image_.x, weights=image_.data)
                stats_.y_bar = np.average(image_.y, weights=image_.data)
                stats_.x_var = np.average((image_.x - stats_.x_bar) ** 2,
                                          weights=image_.data)
                stats_.y_var = np.average((image_.y - stats_.y_bar) ** 2,
                                          weights=image_.data)
                stats_.covariance = np.average(image_.x * image_.y,
                                               weights=image_.data) - \
                                    stats_.x_bar * stats_.y_bar

            # if there is a 'ZeroDivisionError', then delete group
            except ZeroDivisionError:
                del group
                continue

            # if the variance of X or Y is 0, then delete group
            if 0 in [stats_.x_var, stats_.y_var]:
                del group
                continue

            # otherwise, compute rho, covariance matrix
            stats_.rho = stats_.covariance / (np.sqrt(stats_.x_var) *
                                              np.sqrt(stats_.y_var))
            stats_.covariance_matrix = np.array([[stats_.x_var,
                                                  stats_.covariance],
                                                 [stats_.covariance,
                                                  stats_.y_var]])

            # compute statistics required for ellipse parameters
            stats_.eigen_values, stats_.eigen_vectors = \
                np.linalg.eig(stats_.covariance_matrix)

            if stats_.eigen_values[0] >= stats_.eigen_values[1]:
                stats_.radians = np.arctan(stats_.eigen_vectors[1][0] /
                                           stats_.eigen_vectors[1][1])
                stats_.degrees = np.degrees(stats_.radians)
            else:
                stats_.radians = np.arctan(stats_.eigen_vectors[0][0] /
                                           stats_.eigen_vectors[0][1])
                stats_.degrees = np.degrees(stats_.radians)


if __name__ == "__main__":
    import numpy as np

    image = np.zeros((10, 10))
    image[1:4, 1:4] = 1
    image[1:4, 6:9] = 1
    image[6:9, 1:4] = 1
    image[6:9, 6:9] = 1
    cd = ClustarData(image)
    print(cd.image.data)
