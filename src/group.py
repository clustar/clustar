from scipy import stats
from scipy import ndimage
import numpy as np


def _arrange_groups(nonzero, buffer_size, group_size, group_factor):
    # buffer size for group ranges
    b = buffer_size

    # number of nonzero points per group
    groups = [[]]

    # group ranges in the format: ['row min', 'row max',
    # 'col min', 'col max']
    group_ranges = []

    # for each 'row' and 'col' from nonzero indices,
    for row, col in nonzero:

        # boolean flag for optimization
        is_classified = False

        # if there are no defined group ranges, then populate the
        # 'group_ranges' and 'groups' arrays with the first nonzero
        # 'row', 'col' indices
        if len(group_ranges) == 0:
            group_ranges.append([row, row, col, col])
            groups[0].append([row, col])
            continue

        # for each group range in 'group_ranges', check if 'row' and 'col'
        # fall within the group range (+/-) buffer size,
        for i, group_range in enumerate(group_ranges):
            row_min, row_max, col_min, col_max = group_range
            if (row_min - b <= row <= row_max + b) and \
                    (col_min - b <= col <= col_max + b):

                # update min/max of group ranges if 'row' and/or 'col'
                # exceed min/max bounds
                if row_min > row: row_min = row
                if row_max < row: row_max = row
                if col_min > col: col_min = col
                if col_max < col: col_max = col

                # update 'group_ranges' row/col bounds
                group_ranges[i] = [row_min, row_max, col_min, col_max]

                # add 'row', 'col' index to 'groups' array
                groups[i].append([row, col])

                # break loop if 'row', 'col' are accounted
                is_classified = True
                break

        # if 'row', 'col' do not fall within group ranges (+/-) buffer
        # size, then create a new group range; add index to 'groups'
        if not is_classified:
            group_ranges.append([row, row, col, col])
            groups.append([[row, col]])

    # 'group_ranges' may contain ranges that can be nested; for example,
    # the range ['row min'=25, 'row max'=75, 'col min'=25, 'col max'=75]
    # lies within the range ['row min'=0, 'row max'=100, 'col min'=0,
    # 'col max'=100]

    # merge groups that lie within other group ranges
    i = 0
    while i < len(group_ranges):
        row_min, row_max, col_min, col_max = group_ranges[i]
        j = i + 1
        while j < len(group_ranges):
            r_min, r_max, c_min, c_max = group_ranges[j]
            if ((row_min <= r_min <= row_max) and
                    (row_min <= r_max <= row_max) and
                    (col_min <= c_min <= col_max) and
                    (col_min <= c_max <= col_max)):
                groups[i] += groups[j]
                del group_ranges[j]
                del groups[j]
            j += 1
        i += 1

    # filter out small groups that do not meet 'group_size' threshold
    group_ranges = [group_range for group, group_range in
                    zip(groups, group_ranges)
                    if len(group) >= group_size]

    # filter out small groups that do not meet 'group_factor' threshold
    threshold = int(group_factor *
                    np.max([len(group) for group in groups]))
    group_ranges = [group_range for group, group_range in
                    zip(groups, group_ranges)
                    if len(group) >= threshold]

    return group_ranges


def _compute_stats(image, group_ranges):
    # group statistics, data stored as instances of dictionaries
    group_stats = []
    group_data = []

    i = 0
    while i < len(group_ranges):

        # dictionaries to store group statistics and group data
        stats_, data_ = {}, {}

        row_min, row_max, col_min, col_max = group_ranges[i]

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

        # compute elementary statistics; weight by image data
        try:
            x_bar = np.average(x, weights=image_data)
            y_bar = np.average(y, weights=image_data)
            x_var = np.average((x - x_bar) ** 2, weights=image_data)
            y_var = np.average((y - y_bar) ** 2, weights=image_data)
            cov = np.average(x * y, weights=image_data) - x_bar * y_bar

        # if there is a 'ZeroDivisionError', then delete group range
        except ZeroDivisionError:
            # print("\nG{0}: ZeroDivisionError\n".format(i + 1))
            del group_ranges[i]
            continue

        # if the variance of X or Y is 0, then delete group range
        if 0 in [x_var, y_var]:
            # print("\nG{0}: Detected 0 Variance\n".format(i + 1))
            del group_ranges[i]
            continue

        # otherwise, compute rho, covariance matrix
        rho = cov / (np.sqrt(x_var) * np.sqrt(y_var))
        cov_mat = np.array([[x_var, cov], [cov, y_var]])

        # compute statistics required for ellipse parameters
        eig_values, eig_vectors = np.linalg.eig(cov_mat)
        if eig_values[0] > eig_values[1]:
            x_len = 2 * np.sqrt(eig_values[0] *
                                stats.chi2.ppf(1 - 0.2, df=2))
            y_len = 2 * np.sqrt(eig_values[1] *
                                stats.chi2.ppf(1 - 0.2, df=2))
            rad = np.arctan(eig_vectors[1][0] / eig_vectors[1][1])
        else:
            x_len = 2 * np.sqrt(eig_values[1] *
                                stats.chi2.ppf(1 - 0.2, df=2))
            y_len = 2 * np.sqrt(eig_values[0] *
                                stats.chi2.ppf(1 - 0.2, df=2))
            rad = np.arctan(eig_vectors[0][0] / eig_vectors[0][1])

        # store group statistics in a dictionary
        stats_['X_BAR'] = x_bar
        stats_['Y_BAR'] = y_bar
        stats_['COV_MAT'] = cov_mat
        stats_['RHO'] = rho
        stats_['EIG_VALUES'] = eig_values
        stats_['EIG_VECTORS'] = eig_vectors
        stats_['X_LEN'] = x_len
        stats_['Y_LEN'] = y_len
        stats_['RAD'] = rad

        # store group data in a dictionary
        data_['X'] = x
        data_['Y'] = y
        data_['IMAGE'] = image_data
        data_['REF'] = [col_min, row_min]
        data_['LIMIT'] = [image.shape[0], image.shape[1]]

        group_data.append(data_)
        group_stats.append(stats_)
        i += 1

    return group_stats, group_data


def _arrange_subgroups(group_ranges, group_data, group_stats, subgroup_factor):
    group_ranges_ = []

    for range_, data_, stats_ in zip(group_ranges, group_data,
                                     group_stats):

        # rotate image along the major axis as defined by 'stats_['RAD']'
        r = ndimage.rotate(data_['IMAGE'], np.degrees(stats_['RAD']))

        n_rows = r.shape[0]
        n_cols = r.shape[1]

        # row, col indexes for maximum intensity from image data
        r_mid, c_mid = np.where(r == np.max(r))
        r_mid, c_mid = r_mid[0], c_mid[0]

        # image data from the center row of the rotated image
        y = np.array([r[r_mid, c] for c in range(n_cols)])

        # trim off 10% of the image data from both ends
        trim = int(len(y) * 0.10)

        x = np.arange(0 + trim, len(y) - trim, 1)
        y = y[trim:len(y) - trim]

        # compute the first derivative on the image data
        try:
            dydx = np.diff(y) / np.diff(x)

        # ignore non-differentiable groups
        except ValueError:
            print('\nNon-Differentiable Group')
            group_ranges.append(range_)
            continue

        # indexes of critical points (i.e. where the first derivative
        # (dydx) crosses 0 from +/- or -/+)
        indexes = [i for i in range(1, len(dydx))
                   if (dydx[i - 1] > 0 >= dydx[i]) or
                   (dydx[i - 1] < 0 <= dydx[i])]
        values = []

        # if there is more than one critical point, this may indicate the
        # presence of subgroups within the image data
        if len(indexes) > 1:
            # values of critical points along the center row of image data
            values = [r[r_mid, x[indexes[i]]] for i in range(len(indexes))]

            # peak-to-peak amplitude of the values (intensities)
            dists = [r[r_mid, x[indexes[i - 1]]] - r[r_mid, x[indexes[i]]]
                     for i in range(1, len(indexes))]

            # drop values whose amplitudes do not fall within the specified
            # range of the absolute amplitude
            buffer = subgroup_factor * np.max(np.abs(dists))
            limit = np.max(values) - buffer
            values = [value for value in values if value >= limit]

        # remaining values correspond to the centers of a new group
        if len(values) > 1:

            # coords of the critical points on the rotated image
            c = np.array([[x[i], r_mid] for i in indexes])

            # original center of the un-rotated image
            org_center = (np.array(data_['IMAGE'].shape[:2][::-1]) - 1) / 2

            # rotation center of the rotated image
            rot_center = (np.array(r.shape[:2][::-1]) - 1) / 2

            # rotation matrix for rotating image back to original
            r_mat = np.array([[np.cos(stats_['RAD']),
                               np.sin(stats_['RAD'])],
                              [-np.sin(stats_['RAD']),
                               np.cos(stats_['RAD'])]])

            # coords of the critical points on the original image
            coords = np.dot(c - rot_center, r_mat) + org_center

            # Euclidean distance between local maxima and local minima
            buffers = [int(np.linalg.norm(coords[i] - coords[i - 1]))
                       for i in range(1, len(coords))]

            # convert distances into 'group ranges'
            bounds = [np.repeat(coords[i], 2)
                      for i in range(0, len(coords), 2)]
            bounds = np.array(bounds, dtype=int).tolist()

            row_min, col_min = data_['REF']
            r_limit, c_limit = data_['LIMIT']
            for bound, b in zip(bounds, buffers):
                r_min, r_max, c_min, c_max = bound
                r_min += row_min - b
                r_max += row_min + b
                c_min += col_min - b
                c_max += col_min + b

                if r_min < 0: r_min = 0
                if c_min < 0: c_min = 0
                if r_max > r_limit: r_max = r_limit
                if c_max > c_limit: c_max = c_limit

                group_ranges_.append([c_min, c_max, r_min, r_max])
        else:
            group_ranges_.append(range_)
    return group_ranges_


def _convert_to_square_matrix(group_data, group_ranges):
    # converts image data from the groups to a square matrix by inserting
    # additional rows or columns to the shorter axis
    for i, data_ in enumerate(group_data):
        image = data_['IMAGE']
        n_rows = image.shape[0]
        n_cols = image.shape[1]

        # continue for image data that is already a square matrix
        if n_rows == n_cols:
            continue

        diff = np.abs(n_rows - n_cols)
        split = diff // 2
        shape = n_rows if n_rows > n_cols else n_cols

        a = np.zeros((split, shape))
        b = np.zeros((diff - split, shape))

        # update group ranges for new dimensions of the image data
        row_min, row_max, col_min, col_max = group_ranges[i]

        # add roughly equal sets of arrays containing 0's to either ends
        if n_rows > n_cols:
            image = np.insert(image, 0, a, axis=1)
            image = np.insert(image, len(image[0]), b, axis=1)
            col_min -= a.shape[0]
            col_max += b.shape[0]
        else:
            image = np.insert(image, 0, a, axis=0)
            image = np.append(image, b, axis=0)
            row_min -= a.shape[0]
            row_max += b.shape[0]

        group_data[i]['IMAGE'] = image
        group_ranges[i] = [row_min, row_max, col_min, col_max]

    return group_data, group_ranges


def arrange(image, buffer_size, group_size,
            subgroup_factor=0.5, group_factor=0.0):
    # set nan values to 0
    image = np.nan_to_num(image, nan=0)

    # get indices of nonzero elements
    nonzero = np.dstack(np.nonzero(image))[0]

    # determine group ranges from nonzero indices
    group_ranges = _arrange_groups(nonzero, buffer_size,
                                   group_size, group_factor)

    # compute statistics from group ranges
    group_stats, group_data = _compute_stats(image, group_ranges)

    # update group ranges after considering subgroups
    group_ranges = _arrange_subgroups(group_ranges, group_data,
                                      group_stats, subgroup_factor)

    # update statistics from group ranges
    group_stats, group_data = _compute_stats(image, group_ranges)

    # convert group data into square matrices
    group_data, group_ranges = _convert_to_square_matrix(group_data,
                                                         group_ranges)

    return group_ranges, group_data, group_stats
