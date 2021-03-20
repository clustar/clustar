from astropy.io import fits
import numpy as np
import sys
from core import ClustarData


class Clustar(object):

    def __init__(self, **kwargs):
        self.params = kwargs

    def run(self, files):
        output = []
        errors = []
        for i, file_path in enumerate(files):
            file = fits.open(file_path)
            origin_image = file[0].data[0, 0, :, :]
            image = np.array(origin_image, copy=True)

            if image.shape[0] > 2000:
                continue

            try:
                cd = ClustarData(image, **self.params)

                jsn_list = []
                group_num = 1
                for group in cd.groups:
                    jsn = {}
                    jsn['file'] = file_path
                    jsn['group'] = group_num
                    jsn['bounds'] = group.image.bounds
                    jsn['data'] = group.image.data.tolist()
                    jsn['flag'] = cd.flag
                    jsn_list.append(jsn)
                    group_num += 1

            except Exception:
                errors.append(file_path)

            sys.stderr.write(f'\rFile: {i + 1}/{len(files)} ' +
                             f'| Flagged: {len(output)} ' +
                             f'| Errors: {len(errors)}')
            sys.stderr.flush()

        return jsn_list, errors
