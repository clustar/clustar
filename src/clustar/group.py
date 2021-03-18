from scipy import stats
from scipy import ndimage
import numpy as np
import graph


def arrange(cd):
    # get indices of nonzero elements
    nonzero = cd.image.nonzero

    # buffer size for group ranges
    b = cd.params.buffer_size

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

    cd.groups = [cd.Group(bound) for bound in group_ranges]
    return cd


def extract(cd):
    for group in cd.groups:
        row_min, row_max, col_min, col_max = group.image.bounds
        x = np.arange(col_min, col_max + 1, 1)
        y = np.arange(row_min, row_max + 1, 1)
        x, y = np.meshgrid(x, y)
        pos = np.dstack((y, x))

        n_rows = pos.shape[0]
        n_cols = pos.shape[1]

        group.image.data = np.zeros((n_rows, n_cols))
        group.image.clean = np.zeros((n_rows, n_cols))
        for r in range(n_rows):
            for c in range(n_cols):
                group.image.data[r, c] = cd.image.data[tuple(pos[r, c])]
                group.image.clean[r, c] = cd.image.clean[tuple(pos[r, c])]

        x = np.arange(0, n_cols, 1)
        y = np.arange(0, n_rows, 1)

        group.image.x, group.image.y = np.meshgrid(x, y)
        group.image.pos = np.dstack((group.image.x, group.image.y))
        group.image.nonzero = np.dstack(np.nonzero(group.image.clean))[0]
        group.image.ref = [col_min, row_min]
        group.image.limit = [cd.image.data.shape[0],
                             cd.image.data.shape[1]]
    return cd


def calculate(cd):
    for group in cd.groups:
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
        alpha = cd.params.alpha
        if stats_.eigen_values[0] >= stats_.eigen_values[1]:
            stats_.x_len = 2 * np.sqrt(stats_.eigen_values[0] *
                                       stats.chi2.ppf(1 - alpha, df=2))
            stats_.y_len = 2 * np.sqrt(stats_.eigen_values[1] *
                                       stats.chi2.ppf(1 - alpha, df=2))
            stats_.radians = np.arctan(stats_.eigen_vectors[1][0] /
                                       stats_.eigen_vectors[1][1])
            stats_.degrees = np.degrees(stats_.radians)
        else:
            stats_.x_len = 2 * np.sqrt(stats_.eigen_values[1] *
                                       stats.chi2.ppf(1 - alpha, df=2))
            stats_.y_len = 2 * np.sqrt(stats_.eigen_values[0] *
                                       stats.chi2.ppf(1 - alpha, df=2))
            stats_.radians = np.arctan(stats_.eigen_vectors[0][0] /
                                       stats_.eigen_vectors[0][1])
            stats_.degrees = np.degrees(stats_.radians)
    return cd


def rectify(cd):
    # converts image data from the groups to a square matrix by inserting
    # additional rows or columns to the shorter axis
    for group in cd.groups:
        image = group.image.data
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
        row_min, row_max, col_min, col_max = group.image.bounds

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

        group.image.data = image
        group.image.bounds = [row_min, row_max, col_min, col_max]

    return cd


def merge(cd):
    b = cd.params.buffer_size
    group_ranges = [group.image.bounds for group in cd.groups]
    group_ranges = sorted([(range_[1] - range_[0], range_)
                           for range_ in group_ranges], reverse=True)
    group_ranges = [range_[1] for range_ in group_ranges]

    i = 0
    while i < len(group_ranges):
        row_min, row_max, col_min, col_max = group_ranges[i]
        j = i + 1
        while j < len(group_ranges):
            r_min, r_max, c_min, c_max = group_ranges[j]
            if ((row_min - b <= r_min <= row_max + b) and
                    (row_min - b <= r_max <= row_max + b) and
                    (col_min - b <= c_min <= col_max + b) and
                    (col_min - b <= c_max <= col_max + b)):
                del group_ranges[j]
                j -= 1
            j += 1
        i += 1

    cd.groups = [cd.Group(bound) for bound in group_ranges]
    return cd


def refine(cd):
    threshold = cd.params.group_factor * np.max([len(group.image.nonzero)
                                                 for group in cd.groups])
    i = 0
    while i < len(cd.groups):
        if ((len(cd.groups[i].image.nonzero) < cd.params.group_size) or
                (len(cd.groups[i].image.nonzero) < threshold)):
            del cd.groups[i]
            i -= 1
        i += 1

    return cd


def detect(cd):
    group_ranges = []

    for group in cd.groups:
        image_ = group.image
        stats_ = group.stats

        # rotate image along the major axis as defined by 'stats_['RAD']'
        r = ndimage.rotate(image_.data, stats_.degrees)

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
            group_ranges.append(group.image.bounds)
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
            buffer = cd.params.subgroup_factor * np.max(np.abs(dists))
            limit = np.max(values) - buffer
            values = [value for value in values if value >= limit]

        # remaining values correspond to the centers of a new group
        if (1 < len(values) < 3):

            # coords of the critical points on the rotated image
            c = np.array([[x[i], r_mid] for i in indexes])

            # original center of the un-rotated image
            org_center = (np.array(image_.data.shape[:2][::-1]) - 1) / 2

            # rotation center of the rotated image
            rot_center = (np.array(r.shape[:2][::-1]) - 1) / 2

            # rotation matrix for rotating image back to original
            r_mat = np.array([[np.cos(stats_.radians),
                               np.sin(stats_.radians)],
                              [-np.sin(stats_.radians),
                               np.cos(stats_.radians)]])

            # coords of the critical points on the original image
            coords = np.dot(c - rot_center, r_mat) + org_center

            # Euclidean distance between local maxima and local minima
            buffers = [int(np.linalg.norm(coords[i] - coords[i - 1]))
                       for i in range(1, len(coords))]

            # convert distances into 'group ranges'
            bounds = [np.repeat(coords[i], 2)
                      for i in range(0, len(coords), 2)]
            bounds = np.array(bounds, dtype=int).tolist()

            row_min, col_min = image_.ref
            r_limit, c_limit = image_.limit
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

                group_ranges.append([c_min, c_max, r_min, r_max])
        else:
            group_ranges.append(image_.bounds)

    cd.groups = [cd.Group(bound) for bound in group_ranges]
    return cd
