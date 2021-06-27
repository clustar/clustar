from setuptools import setup, find_packages


with open('README.md', 'r') as file:
    long_description = file.read()

setup(
    name='clustar',
    author='Pavan Kumar Bondalapati',
    author_email='pb7ak@virginia.edu',
    description='''A python package for processing and analyzing protostars/
                   protoplanetary disks in astronomical data in Flexible Image
                   Transport System (FITS) images.''',
    version='1.2.1',
    keywords='cluster, astronomy, protostars',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/clustar/clustar', 
    classifiers=[  
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
    package_dir={'': 'src'},  
    packages=find_packages(where='src'),  # Required
    python_requires='>=3.6, <4',
    install_requires=[
        'numpy ~= 1.20.1',
        'scipy ~= 1.6.1',
        'astropy ~= 4.2',
        'matplotlib ~= 3.3.4',
        'pandas ~= 1.2.3',
        'tqdm ~= 4.50.2'
    ],
    extras_require={
        'dev': [
            'pytest >= 5.4.1'
        ]
    },
)