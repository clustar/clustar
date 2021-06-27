from clustar import denoise
from clustar.base import ClustarBase
from scipy import stats
import numpy as np


def test_crop_radius_factor():
    cd = ClustarBase()
    params = [[500, 500, 50, 50, 0, 5]]
    cd.generate_image(params, n_rows=1000, n_cols=1000)
    cd = denoise.crop(cd, radius_factor=0.85)

    radius = cd.image.clean.shape[0] // 2
    marker = int(radius - 0.85 * radius)
    inside = cd.image.clean[radius, marker]
    outside = cd.image.clean[radius, marker-1]
    assert np.isnan(inside) == False
    assert np.isnan(outside) == True
