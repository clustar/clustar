from scipy import stats
from scipy import ndimage
import numpy as np
from shapely import geometry as geom
from shapely import affinity


def _extract_groups(image, group_ranges, group_data):
    for i, range_ in enumerate(group_ranges):
        row_min, row_max, col_min, col_max = range_

        # transform group range into coordinate matrices
        x = np.arange(col_min, col_max + 1, 1)
        y = np.arange(row_min, row_max + 1, 1)
        x, y = np.meshgrid(x, y)

        # position array that contains row/col indexes as ['row', 'col']
        pos = np.dstack((y, x))

        n_rows = pos.shape[0]
        n_cols = pos.shape[1]

        # store image data from position indices specified by group range
        image_data = np.zeros((n_rows, n_cols))
        for r in range(n_rows):
            for c in range(n_cols):
                image_data[r, c] = image[tuple(pos[r, c])]

        group_data[i]['RAW'] = image_data

    return group_data


def _compute_fit(group_ranges, group_data, group_stats):
    residuals = []
    for i, (range_, data_, stats_) in \
            enumerate(zip(group_ranges, group_data, group_stats)):
        fit_stats, fit_data = {}, {}
        dim = data_['RAW'].shape
        x = np.arange(0, dim[1], 1)
        y = np.arange(0, dim[0], 1)
        x, y = np.meshgrid(x, y)

        x_bar = np.average(x, weights=data_['RAW'])
        y_bar = np.average(y, weights=data_['RAW'])
        x_var = np.average((x - x_bar) ** 2, weights=data_['RAW'])
        y_var = np.average((y - y_bar) ** 2, weights=data_['RAW'])
        cov = np.average(x * y, weights=data_['RAW']) - x_bar * y_bar
        cov_mat = np.array([[x_var, cov], [cov, y_var]])

        rv = stats.multivariate_normal([x_bar, y_bar], cov_mat)
        pos = np.dstack((x, y))
        bvg = rv.pdf(pos)

        data_max = np.max(data_['RAW'].ravel())
        bvg_max = np.max(bvg.ravel())
        bvg *= data_max / bvg_max

        residual = 1 - (bvg / data_['RAW'])
        residuals.append(residual)

        fit_stats['X_BAR'] = x_bar
        fit_stats['Y_BAR'] = y_bar
        fit_stats['COV_MAT'] = cov_mat

        fit_data['X'] = x
        fit_data['Y'] = y
        fit_data['POS'] = pos
        fit_data['BVG'] = bvg
        fit_data['RES'] = {'RAW': residual}

        group_stats[i]['FIT'] = fit_stats
        group_data[i]['FIT'] = fit_data

    return group_data, group_stats


def _compute_ellipse(group_data, group_stats):
    for i, (data_, stats_) in enumerate(zip(group_data,
                                            group_stats)):
        res = data_['FIT']['RES']['RAW']
        dim = res.shape
        x_bar = group_stats[i]['FIT']['X_BAR']
        y_bar = group_stats[i]['FIT']['Y_BAR']
        x_len = stats_['X_LEN']
        y_len = stats_['Y_LEN']
        rad = stats_['RAD']

        n = 360
        theta = np.linspace(0, np.pi * 2, n)

        a = x_len / 2
        b = y_len / 2
        angle = np.degrees(rad)

        r = a * b / np.sqrt((b * np.cos(theta)) ** 2 +
                            (a * np.sin(theta)) ** 2)
        xy = np.stack([x_bar + r * np.cos(theta),
                       y_bar + r * np.sin(theta)], 1)

        ellipse = affinity.rotate(geom.Polygon(xy), angle, (x_bar, y_bar))
        # ellipse_x, ellipse_y = ellipse.exterior.xy

        rnd = np.array([[i, j] for i in range(dim[0])
                        for j in range(dim[1])])
        res_in = np.array([p for p in rnd
                           if ellipse.contains(geom.Point(p))])
        res_out = np.array([p for p in rnd
                            if not ellipse.contains(geom.Point(p))])

        group_data[i]['FIT']['RES']['IN'] = res_in
        group_data[i]['FIT']['RES']['OUT'] = res_out

    return group_data


def _compute_residual_stats(group_data, group_stats):
    for i, (data_, stats_) in enumerate(zip(group_data, group_stats)):
        res = data_['FIT']['RES']['RAW']
        res_in = data_['FIT']['RES']['IN']

        output = np.abs(res[res_in[:, 1], res_in[:, 0]])
        output[output < 0] = 0
        output[output > 1] = 1

        group_stats[i]['FIT']['VARIANCE'] = np.std(output) ** 2
        group_stats[i]['FIT']['AVERAGE'] = np.mean(output)
        group_stats[i]['FIT']['WEIGHTED AVERAGE'] = \
            np.average(output, weights=data_['RAW'][res_in[:, 1], res_in[:, 0]])
        group_data[i]['FIT']['ELLIPSE'] = output

    return group_data, group_stats


def _peaks_from_axis(image):
    n_rows = image.shape[0]
    n_cols = image.shape[1]
    mid = n_rows // 2

    y = np.array([image[mid, c] for c in range(n_cols)])
    x = np.arange(0, len(y), 1)

    b = 5
    y_avg = []
    for i in range(len(y)):
        if i + b > y.shape[0]:
            b = b - 1
        if b != 0:
            y_avg.append(np.mean(y[i:i + b]))

    y = np.array(y_avg)
    dydx = np.diff(y) / np.diff(x)

    L = np.array([y[i - 1] / y[i] for i in range(1, len(y) // 2) if y[i] != 0])
    L[L < 0.75] = 0
    L_ = np.nonzero(L)[0]
    L_ = [L_[i - 1] for i in range(1, len(L_)) if ((L_[i] - L_[i - 1]) == 1)]

    R = np.array([y[i] / y[i - 1] for i in range(len(y) - 1, len(y) // 2, -1)
                  if y[i - 1] != 0])
    R[R < 0.75] = 0
    R_ = np.nonzero(R)[0]
    R_ = [R_[i - 1] for i in range(1, len(R_)) if ((R_[i] - R_[i - 1]) == 1)]

    dydx_ = dydx[L_[0]:-R_[0]]

    idx = np.array([i for i in range(1, len(dydx_))
                    if (dydx_[i - 1] > 0 >= dydx_[i])
                    or (dydx_[i - 1] < 0 <= dydx_[i])]) + L_[0]

    idx = [idx[i] for i, val in enumerate(idx) if i % 2 == 0]

    return idx


def _count_peaks(group_data, group_stats):
    for i, (data_, stats_) in enumerate(zip(group_data, group_stats)):
        res = data_['FIT']['RES']['RAW']
        res_out = data_['FIT']['RES']['OUT']
        res = np.array(res, copy=True)
        res[res_out[:, 1], res_out[:, 0]] = 0

        rad = stats_['RAD']
        r_major = np.abs(ndimage.rotate(res, np.degrees(rad)))
        r_minor = np.abs(ndimage.rotate(res, np.degrees(rad) + 90))

        major_idx = _peaks_from_axis(r_major)
        minor_idx = _peaks_from_axis(r_minor)

        group_stats[i]['FIT']['MAJOR PEAKS'] = len(major_idx)
        group_stats[i]['FIT']['MINOR PEAKS'] = len(minor_idx)

    return group_stats


def _filter_threshold(metric, threshold, group_stats):
    output = []
    for stats_ in group_stats:
        if stats_['FIT'][metric.upper()] > threshold:
            if ((stats_['FIT']['MAJOR PEAKS'] not in [2, 4]) or
                    (stats_['FIT']['MINOR PEAKS'] not in [2, 4])):
                output.append(True)
            else:
                output.append(False)
        else:
            output.append(False)
    return output


def bivariate_gaussian(image, metric, threshold,
                       group_ranges, group_data, group_stats):
    group_data = _extract_groups(image, group_ranges, group_data)
    group_data, group_stats = _compute_fit(group_ranges, group_data, group_stats)
    group_data = _compute_ellipse(group_data, group_stats)
    group_data, group_stats = _compute_residual_stats(group_data, group_stats)
    group_stats = _count_peaks(group_data, group_stats)
    output = _filter_threshold(metric, threshold, group_stats)
    return group_data, group_stats, output
