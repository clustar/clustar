import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def identify_peaks(image, smoothing=5, clip=0.75):
    n_rows = image.shape[0]
    n_cols = image.shape[1]
    mid = n_rows // 2

    y = np.array([image[mid, c] for c in range(n_cols)])
    x = np.arange(0, len(y), 1)

    y_avg = []
    for i in range(len(y)):
        if i + smoothing > y.shape[0]:
            smoothing -= - 1
        if smoothing != 0:
            y_avg.append(np.mean(y[i:i + smoothing]))

    y = np.array(y_avg)
    dydx = np.diff(y) / np.diff(x)

    lhs = np.array([y[i - 1] / y[i] for i in
                    range(1, len(y) // 2) if y[i] != 0])
    lhs[lhs < clip] = 0
    lhs = np.nonzero(lhs)[0]
    lhs = [lhs[i - 1] for i in range(1, len(lhs))
           if ((lhs[i] - lhs[i - 1]) == 1)]

    rhs = np.array([y[i] / y[i - 1] for i in
                    range(len(y) - 1, len(y) // 2, -1)
                    if y[i - 1] != 0])
    rhs[rhs < clip] = 0
    rhs = np.nonzero(rhs)[0]
    rhs = [rhs[i - 1] for i in
           range(1, len(rhs)) if ((rhs[i] - rhs[i - 1]) == 1)]

    idx = []

    if len(lhs) > 1 and len(rhs) > 1:
        dydx_ = dydx[lhs[0]:-rhs[0]]

        if len(dydx_) > 2:
            idx = np.array([i for i in range(1, len(dydx_))
                            if (dydx_[i - 1] > 0 >= dydx_[i])
                            or (dydx_[i - 1] < 0 <= dydx_[i])]) + lhs[0]

            idx = [idx[i] for i, val in enumerate(idx) if i % 2 == 0]

    return idx
