"""
Clustar module for fitting-related methods.

This module is designed for the 'ClustarData' object. All listed methods take
an input parameter of a 'ClustarData' object and return a 'ClustarData' object
after processing the method. As a result, all changes are localized within the
'ClustarData' object.

Visit <https://clustar.github.io/> for additional information.
"""


from clustar import graph
from scipy import ndimage, stats
from shapely import affinity, geometry 
import numpy as np


def compute_fit(cd):
    """
    Computes the normalized bivariate gaussian fit for the 'Group' objects.
    
    Parameters
    ----------
    cd : ClustarData
        'ClustarData' object required for processing.
    
    Returns
    -------
    ClustarData
    """
    i = 0
    while i < len(cd.groups):
        group = cd.groups[i]
        try:
            rv = stats.multivariate_normal([group.stats.x_bar,
                                            group.stats.y_bar],
                                           group.stats.covariance_matrix)
        except ValueError:
            del cd.groups[i]
            continue
            
        bvg = rv.pdf(group.image.pos)
        bvg *= np.max(group.image.data) / np.max(bvg)
        
        group.res.data = 1 - (bvg / group.image.data)
        group.fit.bvg = bvg
        group.fit.rv = rv
        i += 1
    return cd

def compute_ellipse(cd):
    """
    Computes the ellipse parameters and localized residuals for the 'Group'
    objects.
    
    Parameters
    ----------
    cd : ClustarData
        'ClustarData' object required for processing.
    
    Returns
    -------
    ClustarData
    """
    for group in cd.groups:
        a = group.stats.x_len / 2
        b = group.stats.y_len / 2

        theta = np.linspace(0, np.pi * 2, 360)

        r = a * b / np.sqrt((b * np.cos(theta)) ** 2 +
                            (a * np.sin(theta)) ** 2)
        xy = np.stack([group.stats.x_bar + r * np.cos(theta),
                       group.stats.y_bar + r * np.sin(theta)], 1)

        ellipse = affinity.rotate(geometry.Polygon(xy),
                                  group.stats.degrees,
                                  (group.stats.x_bar, group.stats.y_bar))

        pos = np.array([[i, j] for i in range(group.image.data.shape[0])
                        for j in range(group.image.data.shape[1])])
        inside = np.array([p for p in pos
                           if ellipse.contains(geometry.Point(p))])
        outside = np.array([p for p in pos
                            if not ellipse.contains(geometry.Point(p))])

        group.fit.ellipse = ellipse
        group.res.pos = pos
        group.res.inside = inside
        group.res.outside = outside
    return cd

def compute_metrics(cd):
    """
    Computes the evaluation metrics for the 'Group' objects.
    
    Parameters
    ----------
    cd : ClustarData
        'ClustarData' object required for processing.
    
    Returns
    -------
    ClustarData
    """
    for group in cd.groups:
        res = group.res
        output = np.abs(res.data[res.inside[:, 0], res.inside[:, 1]])
        output[output < 0] = 0
        output[output > 1] = 1

        bias = group.image.data[res.inside[:, 0], res.inside[:, 1]]
        group.metrics.standard_deviation = np.std(output)
        group.metrics.variance = group.metrics.standard_deviation ** 2
        group.metrics.average = np.mean(output)
        group.metrics.weighted_average = np.average(output, weights=bias)
        group.res.output = output
    return cd

def compute_peaks(cd):
    """
    Computes the number of peaks along the major and minor axes for the 
    'Group' objects.
    
    Parameters
    ----------
    cd : ClustarData
        'ClustarData' object required for processing.
    
    Returns
    -------
    ClustarData
    """
    for group in cd.groups:
        res = np.array(group.res.data, copy=True)
        res_out = group.res.outside
        res[res_out[:, 0], res_out[:, 1]] = 0
        
        r_major = np.abs(ndimage.rotate(res, group.stats.degrees))
        r_minor = np.abs(ndimage.rotate(res, group.stats.degrees + 90))

        major_idx = graph.critical_points(r_major)
        minor_idx = graph.critical_points(r_minor)

        major_idx = [major_idx[i] for i in range(len(major_idx))
                     if i % 2 == 0]
        minor_idx = [minor_idx[i] for i in range(len(minor_idx))
                     if i % 2 == 0]

        group.fit.major_peaks = len(major_idx)
        group.fit.minor_peaks = len(minor_idx)
        group.res.clean = res
    return cd

def validate(cd):
    """
    Determines which 'Group' objects are flagged for manual review by using
    the specified validation parameters.
    
    Parameters
    ----------
    cd : ClustarData
        'ClustarData' object required for processing.
    
    Returns
    -------
    ClustarData
    """
    attribute = cd.params.metric.lower()
    threshold = cd.params.threshold
    for group in cd.groups:
        metric = getattr(group.metrics, attribute)
        if metric > threshold:
            group.flag = True
            cd.flag = True
            if cd.params.evaluate_peaks and \
            ((group.fit.major_peaks in [2, 4]) or 
             (group.fit.minor_peaks in [2, 4])):
                group.flag = False
                cd.flag = False
    return cd
