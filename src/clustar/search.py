from astropy.io import fits
import numpy as np
import sys
from clustar import core
from clustar.core import ClustarData


class Clustar(object):

    def __init__(self, **kwargs):
        self.params = kwargs
        self.files = []

    def run(self, files):
        counter = 0
        jsn_list = []
        for i, file_path in enumerate(files):
            file = fits.open(file_path)
            origin_image = file[0].data[0, 0, :, :]
            image = np.array(origin_image, copy=True)

            if image.shape[0] > 2000:
                continue

            cd = ClustarData(file_path, **self.params)
            self.files.append(cd)

            for index, group in enumerate(cd.groups):
                jsn = {'file': file_path,
                       'group': index,
                       'bounds': group.image.bounds,
                       'data': group.image.data.tolist(),
                       'residuals': group.res.data.tolist(),
                       'flag': cd.flag}
                jsn_list.append(jsn)

            if cd.flag:
                counter += 1

            sys.stderr.write(f'\rFile: {i + 1}/{len(files)} ' +
                             f'| Flagged: {counter} ')
            sys.stderr.flush()

        return jsn_list
