from setuptools import find_packages, setup

# Read README for long description
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='gw_remnant',
    version='0.2.0',  # Updated version
    author='Tousif Islam, Scott Field, Gaurav Khanna',
    author_email='tousifislam24@gmail.com',
    maintainer='Tousif Islam',
    maintainer_email='tousifislam24@gmail.com',
    description='Python package to extract remnant black hole properties from gravitational waveforms',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tousifislam/gw_remnant',
    project_urls={
        'Bug Reports': 'https://github.com/tousifislam/gw_remnant/issues',
        'Source': 'https://github.com/tousifislam/gw_remnant',
    },
    packages=find_packages(),  # Automatically finds all packages
    license='MIT',
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'matplotlib>=3.3.0',
        'gwtools',  # Add if this is required
    ],
    extras_require={
        'surrogates': [
            'gwsurrogate',
            'surfinBH',
        ],
        'lal': [
            'lal',
            'lalsimulation',
        ],
        'all': [
            'gwsurrogate',
            'surfinBH',
            'lal',
            'lalsimulation',
        ],
        'dev': [
            'pytest>=6.0',
            'pytest-cov',
            'black',
            'flake8',
            'ipython',
            'jupyter',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Other Audience',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Operating System :: OS Independent',
    ],
    keywords='gravitational waves, black holes, numerical relativity, remnant properties',
)