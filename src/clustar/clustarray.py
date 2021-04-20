import warnings
import numpy as np
import itertools
from clustar import *
class ClustArray:
    ''' Class for working with data from FITS images
        Initialized from a numpy array from an image
        Methods for denoising images
    '''

    def __init__(self, np_array):
        self.im_array = np_array
        self.noise_est = None
        self.denoised_arr = None

    def circle_crop(self, rad_factor = 1.0):
        '''Function to crop square images to a circle

        Params
        ------
        rad_factor: float multiple allowing change to size of circle_crop
            default is 1
            value equal to 0.7 crops to a circle with radius that is 70% as large as the max image radius
            values < 0 not allowed
            values >= sqrt(2) will return original image

        Outputs
        -------
        new_imdata: np array of same size as image data array, but with values outside radius set to nan;
            sets self.denoised_arr to equal this array
        '''

        if rad_factor < 0:
            raise ValueError('rad_factor must be >= 0')

        if self.denoised_arr is None:
            new_imdata = self.im_array.copy()
        else:
            new_imdata = self.denoised_arr.copy()

        rad = (new_imdata.shape[0]/2)
        rad_sq = (rad*rad_factor)**2

        for ix,iy in np.ndindex(new_imdata.shape):
            if (ix - rad)**2 + (iy - rad)**2 > rad_sq:
                new_imdata[ix, iy] = np.nan

        self.denoised_arr = new_imdata

        return new_imdata

    def pb_multiply(self, pb_array):
        '''Function to multiply a FITS image by a .pb file to deemphasize edges

        Inputs
        ------
        pb_array: numpy array from a .pb file

        Outputs
        -------
        new_imdata: np array of same size as image data array
            consisting of elementwise multiplication of image and pb file;
            sets self.denoised_arr to equal this array
        '''

        if self.denoised_arr is None:
            imdata = self.im_array.copy()
        else:
            imdata = self.denoised_arr.copy()

        new_imdata = np.multiply(imdata, pb_array)

        self.denoised_arr = new_imdata

        return new_imdata

    def get_noise_level(self, nchunks = 3, rms_quantile = 0):
        '''Calculates estimated noise level in image intensity
        Stores value in FitsImage object noise attribute

        Arguments
        ---------
        nchunks: int number of chunks to use in grid, must be odd
        rms_quantile: float in range [0, 1] indicating quantile of chunk RMS to use for noise level (0 = min RMS, 0.5 = median, etc)

        Returns
        -------
        noise: float estimated noise in image intensity values;
            sets self.noise_est to this value
        '''

        if self.denoised_arr is None:
            imdata = self.im_array.copy()
            warnings.warn('Calculating noise level from uncleaned image')
        else:
            imdata = self.denoised_arr.copy()

        #now break the image into chunks and do the same analysis;
        # one of the chunks should have no signal in and give you an estimate of the noise (= rms).# number of chunks in each direction:
        # an odd value is used so that the centre of the image does not correspond to the edge of chunks;
        # when you ask for observations with ALMA, you usually specify that the object of interest be in the
        # center of your image.
        size = [i//nchunks for i in imdata.shape]
        remain = [i % nchunks for i in imdata.shape]
        chunks = dict()
        k = 0
        for j,i in itertools.product(range(nchunks),range(nchunks)):
            chunks[k] = size.copy()
            k += 1# next, account for when the image dimensions are not evenly divisible by `nchunks`.
        row_remain, column_remain = 0, 0
        for k in chunks:
            if k % nchunks < remain[0]:
                row_remain = 1
            if k // nchunks < remain[1]:
                column_remain = 1
            if row_remain > 0:
                chunks[k][0] += 1
                row_remain -= 1
            if column_remain > 0:
                chunks[k][1] += 1
                column_remain -= 1# with that in hand, calculate the lower left corner indices of each chunk
        indices = dict()
        for k in chunks:
            indices[k] = chunks[k].copy()
            if k % nchunks == 0:
                indices[k][0] = 0
            elif k % nchunks != 0:
                indices[k][0] = indices[k-1][0] + chunks[k][0]
            if k >= nchunks:
                indices[k][1] = indices[k-nchunks][1] + chunks[k][1]
            else:
                indices[k][1] = 0
        stddev_chunk = dict()
        rms_chunk = dict()
        for k in chunks:
            i,j = indices[k]
            di,dj = chunks[k]
            x = imdata[i:i+di,j:j+dj]
            stddev_this = np.nanstd(x)
            rms_this = np.sqrt(np.nanmean(x**2))
            stddev_chunk[k] = stddev_this
            rms_chunk[k] = rms_this

        noise = np.quantile(list(rms_chunk.values()), q = rms_quantile)
        self.noise_est = noise
        return(noise)

    def denoise(self, pb_array = None, rad_factor = 1.0, rms_quantile = 0, grid_chunks = 3):
        '''Wrapper function to perform entire denoising process
        Crops image to a circle, multiplies by a pb file (if desired), and calculates RMS noise level

        Inputs
        ------
        im_array: 2d array representing a FITS image data
        pb_array: optional numpy array from a .pb file

        Params
        ------
        rad_factor: float multiple allowing change to size of circle_crop
            default is 1
            value equal to 0.7 crops to a circle with radius that is 70% as large as the max image radius
            values < 0 not allowed
            values >= sqrt(2) will return original image
        grid_chunks: int number of chunks to use in grid, must be odd
        rms_quantile: float in range [0, 1] indicating quantile of chunk RMS to use for noise level (0 = min RMS, 0.5 = median, etc)

        Outputs
        -------
        '''

        self.circle_crop(rad_factor)

        if pb_array is not None:
            self.pb_multiply(pb_array)

        noise_lvl = self.get_noise_level()

        return(noise_lvl)

    def extract_subgroup(self, group_indices, square = True, buffer = 0.0):
        '''Function for extracting a subgroup of an image

            Inputs
            ------
            group_indices: list containing indices of subgroup [row_min, row_max, col_min, col_max]

            Params
            ------
            square: if True, widen shorter axis range to make subgroup a square
            buffer: fraction to add to each dimension
                (e.g. if subgroup is 200x200 pixels, buffer = 0.1 will return 220x220 pixels)
        '''
        row_min = group_indices[0]
        row_max = group_indices[1]
        col_min = group_indices[2]
        col_max = group_indices[3]

        if square:
            diff = (row_max - row_min) - (col_max - col_min)

            if diff == 0:
                #already square
                pass
            elif diff < 0:
                #adjust row min/max
                row_min += int(np.floor(diff/2))
                row_max -= int(np.ceil(diff/2))
            else:
                #adjust col min/max
                col_min -= int(np.floor(diff/2))
                col_max += int(np.ceil(diff/2))

        buffer_width = int(buffer*(col_max - col_min)/2)
        buffer_height = int(buffer*(row_max - row_min)/2)

        row_min -= buffer_height
        row_max += buffer_height
        col_min -= buffer_width
        col_max += buffer_width

        subgroup = self.im_array[row_min:row_max, col_min:col_max]

        return subgroup


    def plot_subgroup(self, group_indices, square = True, buffer = 0.0, colorbar = True):
        '''Function for plotting a subgroup of an image

            Inputs
            ------
            group_indices: list containing indices of subgroup [row_min, row_max, col_min, col_max]

            Params
            ------
            square: if True, widen shorter axis range to make subgroup a square
            buffer: fraction to add to each dimension
                (e.g. if subgroup is 200x200 pixels, buffer = 0.1 will return 220x220 pixels)
            colorbar: boolean indicating whether or not to include a colorbar with the plot
        '''
        subgroup = self.extract_subgroup(group_indices, square, buffer)

        plt.imshow(subgroup, origin='lower')
        if colorbar:
            plt.colorbar()
