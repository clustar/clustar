"""
General module for graphing-related methods.

Visit <https://clustar.github.io/> for additional information.
"""


from scipy import ndimage
from matplotlib.patches import Rectangle, Ellipse
import matplotlib.pyplot as plt
import numpy as np


def critical_points(image, angle=0, smoothing=5, clip=0.75, center=None):
    """
    Returns the number of smoothed critical points along the specified axis.
    
    Parameters
    ----------
    image : ndarray
        Data from the FITS image.
    
    angle : float, optional
        Degree of rotation used to specify the axis of differentiation.
    
    smoothing : int, optional
        Size of window used in the moving average smoothing process for peak
        evaluation.
    
    clip : float, optional
        Determines the percentage of tail values that are trimmed for peak 
        evaluation.
    
    center : int, optional
        Defines the row-wise axis of differentiation for peak evaluation.
    
    Returns
    -------
    list
    """
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
    """
    Displays the FITS image and identifies the groups in green, orange, or
    red rectangles, which are defined as:
    1. Green denotes that the group is not flagged for manual review
    2. Orange denotes that the group is not flagged for manual review, but
       the group is smaller than the beam size.
    3. Red denotes that the group is flagged for manual review.
    Beam size is the white oval shown on the bottom right corner of 
    the FITS image.

    Parameters
    ----------
    cd : ClustarData
        'ClustarData' object required for processing.
    
    vmin : float, optional
        Lower bound for the shown intensities. 

    vmax : float, optional
        Upper bound for the shown intensities.

    show : bool, optional
        Determines whether the groups should be identified. If false, the
        rectangles identifying the groups are not drawn.

    dpi : int, optional
        Dots per inch.
    """
    plt.figure(figsize=(10, 5), dpi=dpi)
    plt.imshow(cd.image.data, origin="lower", vmin=vmin, vmax=vmax)
    plt.title(dict(cd.image.header)['OBJECT'])

    for i, group in enumerate(cd.groups):
        bounds = group.image.bounds
        length = bounds[1] - bounds[0]
        if show:
            edgecolor = "red" if group.flag else "lime"
            warning = len(group.image.nonzero) < cd.image.area
            edgecolor = "orange" if warning else edgecolor
            plt.gca().add_patch(Rectangle(tuple(group.image.ref),
                                          length, length,
                                          edgecolor=edgecolor,
                                          facecolor='none', lw=0.5))

        plt.annotate(i + 1, xy=(bounds[3] + 10, bounds[1] + 10), fontsize=6.5,
                     color='white')
    # show beam size
    major = cd.image.major
    minor = cd.image.minor
    degrees = cd.image.degrees
    row_max, col_max = cd.image.data.shape
    buffer = int(row_max * 0.025)
    coords = ((col_max - minor//2 - buffer), buffer + minor//2)
    plt.gca().add_patch(Ellipse(coords, minor, major, angle=degrees,
                                edgecolor='white', facecolor='white'))
    plt.colorbar()
    plt.show()
