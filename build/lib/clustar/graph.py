import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from clustar import *

def critical_points(image, angle=0, smoothing=5, clip=0.75, center=None):
    # rotate image along the degree of the major axis
    if angle != 0:
        image = ndimage.rotate(image, angle)

    n_rows = image.shape[0]
    n_cols = image.shape[1]
    if center is None:
        mid = n_rows // 2
    else:
        mid = center

    # x and y coordinates along the center horizontal line
    y = np.array([image[mid, c] for c in range(n_cols)])
    x = np.arange(0, len(y), 1)

    # perform the smoothing operation by taking the average of the y values
    # along a moving window as defined by the smoothing parameter
    y_avg = []
    for i in range(len(y)):
        if i + smoothing > y.shape[0]:
            smoothing -= - 1
        if smoothing != 0:
            y_avg.append(np.mean(y[i:i + smoothing]))
    y = np.array(y_avg)

    # determine the first derivative of y against x
    dydx = np.diff(y) / np.diff(x)

    # for the left hand side of the y values...
    # calculate the relative change for each point to the next point
    lhs = np.array([y[i - 1] / y[i] for i in
                    range(1, len(y) // 2) if y[i] != 0])
    lhs[lhs < clip] = 0

    # check if points are roughly consecutive
    lhs = np.nonzero(lhs)[0]
    lhs = [lhs[i - 1] for i in range(1, len(lhs))
           if ((lhs[i] - lhs[i - 1]) == 1)]

    # for the right hand side of the y values...
    # calculate the relative change for each point to the next point
    rhs = np.array([y[i] / y[i - 1] for i in
                    range(len(y) - 1, len(y) // 2, -1)
                    if y[i - 1] != 0])
    rhs[rhs < clip] = 0

    # check if points are roughly consecutive
    rhs = np.nonzero(rhs)[0]
    rhs = [rhs[i - 1] for i in
           range(1, len(rhs)) if ((rhs[i] - rhs[i - 1]) == 1)]

    # obtain coordinates of the critical points
    idx = []
    if len(lhs) > 1 and len(rhs) > 1:
        dydx_ = dydx[lhs[0]:len(dydx) - rhs[0]]

        if len(dydx_) > 2:
            idx = np.array([i for i in range(1, len(dydx_))
                            if (dydx_[i - 1] > 0 >= dydx_[i])
                            or (dydx_[i - 1] < 0 <= dydx_[i])]) + lhs[0]

            idx = np.array([[x[i], mid] for i in idx]).tolist()
            idx.insert(0, [lhs[0], mid])
            idx.append([(len(y) - rhs[0]), mid])

    return idx


def identify_groups(cd, vmin=None, vmax=None, show=True, dpi=180):
    plt.figure(figsize=(10, 5), dpi=dpi)
    plt.imshow(cd.image.data, origin="lower", vmin=vmin, vmax=vmax)
    plt.title(dict(cd.header)['OBJECT'])

    for i, group in enumerate(cd.groups):
        bounds = group.image.bounds
        length = bounds[1] - bounds[0]
        if show:
            edgecolor = "red" if group.flag else "lime"
            plt.gca().add_patch(Rectangle(tuple(group.image.ref),
                                          length, length,
                                          edgecolor=edgecolor,
                                          facecolor='none', lw=0.5))

        plt.annotate(i + 1, xy=(bounds[3] + 10, bounds[1] + 10), fontsize=6.5,
                     color='white')

    # show bean size
    row_max, col_max = cd.image.data.shape
    bean_size = dict(cd.header)['BPA']
    bean_dim = round(abs(bean_size) ** (1 / 2))
    border_size = 3
    border_dim = bean_dim + (2 * border_size)
    border_coords = ((col_max - border_dim - 25), 25)
    bean_coords = ((col_max - border_dim + border_size - 25),
                   25 + border_size)

    plt.gca().add_patch(Rectangle(border_coords,
                                  border_dim, border_dim,
                                  edgecolor='white', facecolor='white'))
    plt.gca().add_patch(Rectangle(bean_coords,
                                  bean_dim, bean_dim, edgecolor='black',
                                  facecolor='black'))

    plt.colorbar()
    plt.show()

