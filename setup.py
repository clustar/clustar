import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='clustar',
    author='NRAO-CapstoneGroup',
    author_email='jz5jx@virginia.edu',
    description='Package for analyzing and clustering astronomical objects',
    keywords='cluster, astronomy, protostars',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/clustar',
    project_urls={
        'Documentation': 'NA',
        'Bug Reports':
        'https://github.com/clustar/clustar/issues',
        'Source Code': 'https://github.com/clustar/clustar/tree/main/src',
        # 'Funding': '',
        # 'Say Thanks!': '',
    },
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    classifiers=[
        # see https://pypi.org/classifiers/
        'Development Status :: 2 - Pre-Alpha',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    # install_requires=['Pillow'],
    include_package_data = True,
    package_data={'': ['data/*.csv']},
    extras_require={
        'dev': ['check-manifest'],
        # 'test': ['coverage'],
    },
    # entry_points={
    #     'console_scripts': [  # This can provide executable scripts
    #         'run=examplepy:main',
    # You can execute `run` in bash to run `main()` in src/examplepy/__init__.py
    #     ],
    # },
)
