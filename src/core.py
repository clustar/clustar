
class ClustarData(object):

    class _Image(object):

        def __init__(self, image):
            self.x = None
            self.y = None
            self.z = image
            self.ref = None
            self.limit = None
            self.pos = None
            self.bvg = None

    class _Residual(object):

        def __init__(self, residuals):
            self.pos = residuals
            self.inside = None
            self.outside = None

    def __init__(self, image, residuals):
        self.image = image
        self.residuals = residuals


class ClustarStats(object):

    def __init__(self, image, group_range):
        self.image = image
        self.group_range = group_range

    def mean(self):
        pass

    def variance(self):
        pass

    def covariance(self):
        pass

    def covariance_matrix(self):
        pass

    def rho(self):
        pass

    def eigen_values(self):
        pass

    def eigen_vectors(self):
        pass

    def angle(self):
        pass

    def ellipse(self):
        pass


if __name__ == "__main__":
    print("Hello")