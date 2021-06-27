"""
Clustar
=======

A python package for processing and analyzing protostars/protoplanetary disks
in astronomical data in Flexible Image Transport System (FITS) images. 

These files contain grayscale images represented as two-dimensional arrays,
with each pixel containing the intensity values, and headers containing the
telescope observational parameters.

Clustar simplifies and expediates the identification pipeline of FITS files
by automating the preprocessing, grouping, and fitting for a large amount of
FITS files.

Singular Usage
--------------
Detect celestial objects in a FITS image by creating a 'ClustarData' object.
>>> from clustar.core import ClustarData

Create the 'ClustarData' object by specifying the path to FITS file. 
>>> cd = ClustarData(path='~/data/example.fits', threshold=0.025)

Visualize the detected groups.
>>> cd.identify()

Access individual 'Group' objects.
>>> cd.groups

Multiple Usage
--------------
Detect celestial objects in a directory containing multiple FITS images.
>>> from clustar.search import Clustar

Setup 'Clustar' object.
>>> cs = Clustar(radius_factor=0.95, threshold=0.025)

Execute pipeline on directory containing FITS files.
>>> cs.run(directory='~/data/')

Access individual 'ClustarData' objects.
>>> cs.data

Check which FITS files raised an error.
>>> cs.errors

Inspect 'ClustarData' variables for all groups in each FITS file.
>>> cs.display(category='all')

Modules
-------
base.py
    Internal module for testing clustar modules.

core.py
    Contains the 'ClustarData' class, which is responsible for executing
    the entire project pipeline for detecting groups in a single FITS image.

denoise.py
    Clustar module for denoising-related methods.

fit.py
    Clustar module for fitting-related methods.

graph.py
    General module for graphing-related methods.

group.py
    Clustar module for grouping-related methods.

search.py
    Contains the 'Clustar' hierarchical class, which is responsible for 
    transforming all available FITS images in a specified directory into their 
    respective 'ClustarData' objects.

Notes
-----
Visit <https://clustar.github.io/> for additional information.
"""
