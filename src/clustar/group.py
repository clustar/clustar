from scipy import stats
from scipy import ndimage
import numpy as np
from clustar import graph


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
        bounds = group.image.bounds
        n_rows, n_cols = [bounds[i] - bounds[i - 1]
                          for i in range(1, len(bounds)) if i % 2 == 1]

        # continue for image data that is already a square matrix
        if n_rows == n_cols:
            continue

        diff = np.abs(n_rows - n_cols)
        split = diff // 2
        shape = n_rows if n_rows > n_cols else n_cols

        # update group ranges for new dimensions of the image data
        row_min, row_max, col_min, col_max = bounds
        r_max, c_max = cd.image.data.shape

        if n_rows > n_cols:
            col_min -= split
            col_min = col_min if col_min > 0 else 0
            col_max += diff - split
            col_max = col_max if col_max < c_max else c_max - 1
        else:
            row_min -= split
            row_min = row_min if row_min > 0 else 0
            row_max += diff - split
            row_max = row_max if row_max < r_max else r_max - 1

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
    if len(cd.groups) > 0:
        group_ranges = [group.image.bounds for group in cd.groups]
        group_sizes = [(group_range[1] - group_range[0]) ** 2
                       for group_range in group_ranges]

        threshold = cd.params.group_factor * np.max(group_sizes)
        i = 0
        while i < len(group_sizes):
            if ((group_sizes[i] < cd.params.group_size) or
                    (group_sizes[i] < threshold)):
                del cd.groups[i]
                del group_sizes[i]
                i -= 1
            i += 1

    return cd


def detect(cd):
    group_ranges = []
    for group in cd.groups:

        # rotate image along the degree of the major axis
        rotated_image = ndimage.rotate(group.image.data, group.stats.degrees)

        # coordinates of the critical points on the rotated image
        idx = graph.critical_points(rotated_image, smoothing=1)
        idx = [idx[i] for i in range(len(idx)) if i % 2 == 0]

        if len(idx) == 3:

            # original center of the un-rotated image
            org_center = (np.array(group.image.data.shape[:2][::-1]) - 1) / 2

            # rotation center of the rotated image
            rot_center = (np.array(rotated_image.shape[:2][::-1]) - 1) / 2

            # rotation matrix for rotating image back to original
            r_mat = np.array([[np.cos(group.stats.radians),
                               np.sin(group.stats.radians)],
                              [-np.sin(group.stats.radians),
                               np.cos(group.stats.radians)]])

            # coordinates of the critical points on the original image
            coords = np.dot(idx - rot_center, r_mat) + org_center
            points = np.array(np.round(coords), dtype=int).tolist()

            # convert coordinates into a group range
            row_min, col_min = group.image.ref
            r_limit, c_limit = group.image.limit
            for i in range(1, len(coords)):
                a = points[i - 1]
                b = points[i]

                r, c = np.dstack([a, b])[0]

                r += row_min
                r = [r_limit if r_ > r_limit else r_ for r_ in r]
                r = [0 if r_ < 0 else r_ for r_ in r]

                c += col_min
                c = [c_limit if c_ > c_limit else c_ for c_ in c]
                c = [0 if c_ < 0 else c_ for c_ in c]
                group_range = [np.min(c), np.max(c), np.min(r), np.max(r)]
                group_ranges.append(group_range)

        else:
            group_ranges.append(group.image.bounds)

        cd.groups = [cd.Group(bound) for bound in group_ranges]

    return cd
