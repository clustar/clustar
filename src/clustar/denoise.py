"""
Clustar module for denoising-related methods.

This module is designed for the 'ClustarData' object. All listed methods take
an input parameter of a 'ClustarData' object and return a 'ClustarData' object
after processing the method. As a result, all changes are localized within the
'ClustarData' object.

Visit <https://clustar.github.io/> for additional information.
"""


from scipy import stats
import concurrent
import itertools
import numpy as np


def crop(cd, radius_factor=1, apply_gradient=True):
    """
    Crops the square FITS image into a circle.
    
    Parameters
    ----------
    cd : ClustarData
        'ClustarData' object required for processing.
        
    radius_factor : float, optional
        Factor multiplied to radius to determine cropping circle in the 
        denoising process; must be within the range [0, 1].
        
    apply_gradient : bool, optional
        Determine if the FITS image should be multiplied by a gradient
        in order to elevate central points; similar to multiplying the
        FITS image by the associated 'pb' data.
    
    Returns
    -------
    ClustarData    
    """    
    assert radius_factor >= 0, "Argument 'radius_factor' must be " + \
    "at least 0."
    
    data = cd.image.data
    output = data.copy()
    if radius_factor > 1:
        cd.image.clean = output
        return cd
    
    pb_X, pb_Y = data.shape
    center = [pb_X/2, pb_Y/2]
    rv = stats.multivariate_normal(center, [[pb_X**2/2, 0], 
                                            [0, pb_Y**2/2]])
    bvg = rv.pdf(cd.image.pos)
    bvg = bvg/np.max(bvg)
    
    radius = center[0] - (center[0] * radius_factor)
    limit = bvg[int(center[0]), int(radius)]
    
    if apply_gradient:
        bvg[bvg < limit] = np.nan
    else:
        bvg = np.where(bvg < limit, np.nan, 1)
    
    output *= bvg
    cd.image.clean = output
    return cd

def _extract_grid(data, n=3):
    size = [i // n for i in data.shape]
    remain = [i % n for i in data.shape]
    chunks = {i:size.copy() for i, (_,_) in 
              enumerate(itertools.product(range(n), range(n)))}
    
    row_remain, column_remain = (0, 0)
    for k in chunks:
        if k % n < remain[0]:
            row_remain = 1
        if k // n < remain[1]:
            column_remain = 1
        if row_remain > 0:
            chunks[k][0] += 1
            row_remain -= 1
        if column_remain > 0:
            chunks[k][1] += 1
            column_remain -= 1
    
    indices = dict()
    for k in chunks:
        indices[k] = chunks[k].copy()
        if k % n == 0:
            indices[k][0] = 0
        elif k % n != 0:
            indices[k][0] = indices[k-1][0] + chunks[k][0]
        if k >= n:
            indices[k][1] = indices[k-n][1] + chunks[k][1]
        else:
            indices[k][1] = 0
    
    chunks = list(chunks.values())
    indices = list(indices.values())
    return chunks, indices

def _compute_stats(index, chunk, data):
    ir, ic = index
    ix, iy = chunk
    x = data[ir:ir+ix, ic:ic+iy]
    std = np.nanstd(x)
    rms = np.sqrt(np.nanmean(x**2))
    return (std, rms)

def compute_noise(cd, n=3, quantile=0.5):
    """
    Computes the noise level by evaluating the root mean square (RMS) metric 
    at the specified quantile on the composed grid.
    
    Parameters
    ----------
    cd : ClustarData
        'ClustarData' object required for processing.
        
    n : int, optional
        Number of chunks to use in a grid; must be an odd number.
    
    quantile : float, optional
        Quantile of RMS to determine the noise level; must be within the 
        range [0, 1].

    Returns
    -------
    ClustarData
    """
    data = cd.image.clean 
    chunks, indices = _extract_grid(data, n)
    
    statistics = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(_compute_stats, indices, chunks, 
                               itertools.repeat(data, len(indices)))
        statistics = [result for result in results]
        
    cd.image.std, cd.image.rms = np.dstack(np.array(statistics))[0]
    cd.image.noise = np.quantile(list(cd.image.rms), q=quantile)
    return cd

def resolve(cd):
    """
    Performs the complete denoising operation on the FITS image.
    
    Parameters
    ----------
    cd : ClustarData
        'ClustarData' object required for processing.

    Returns
    -------
    ClustarData
    """
    
    cd = crop(cd, cd.params.radius_factor, cd.params.apply_gradient)
    cd = compute_noise(cd, cd.params.chunks, cd.params.quantile)
    
    output = cd.image.clean
    output[output < cd.image.noise * cd.params.sigma] = 0
    output = np.nan_to_num(output, nan=0)
    
    cd.image.nonzero = np.dstack(np.nonzero(output))[0]
    cd.image.data = np.nan_to_num(cd.image.data, nan=0)
    cd.image.clean = output
    return cd
