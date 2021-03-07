from astropy.io import fits
import numpy as np
import sys
import group
import fit


class Clustar(object):

    def __init__(self, buffer_size=15, group_size=100, subgroup_factor=0.5,
                 group_factor=0.0, metric="variance", threshold=0.01):
        self.buffer_size = buffer_size
        self.group_size = group_size
        self.subgroup_factor = subgroup_factor
        self.group_factor = group_factor
        self.metric = metric
        self.threshold = threshold
        self.group_ranges = []
        self.group_data = []
        self.group_stats = []

    def run(self, files):
        select = []
        output = []
        for i, file_path in enumerate(files):
            file = fits.open(file_path)
            origin_image = file[0].data[0, 0, :, :]
            image = np.array(origin_image, copy=True)

            if image.shape[0] > 2000:
                continue

            # -- Denoise Here --
            std = np.std(image)
            image[image < std * 5] = 0
            # -- ------------ --

            try:
                self.group_ranges, self.group_data, self.group_stats = \
                    group.arrange(image, self.buffer_size, self.group_size,
                                  self.subgroup_factor, self.group_factor)

                self.group_data, self.group_stats, output = \
                    fit.bivariate_gaussian(origin_image, self.metric,
                                           self.threshold, self.group_ranges,
                                           self.group_data, self.group_stats)

            except Exception:
                select.append(file_path)

            if True in output:
                select.append(file_path)

            sys.stderr.write(f'\rFile: {i + 1}/{len(files)} ' +
                             f'| Flagged: {len(select)}')
            sys.stderr.flush()

        return select
