from setuptools import setup, find_packages
import pathlib
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')
setup(
    name='clustar',
    author='NRAO-CapstoneGroup',
    author_email='jz5jx@virginia.edu',
    description='Package for analyzing and clustering astronomical objects',
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
    package_data={
        "": ["data/*.csv"],
    },
    python_requires='>=3.6, <4',
)
