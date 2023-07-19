from setuptools import find_packages, setup

long_description = open('README.md').read()

setup(name='gw_remnant',
      version='0.0.1',
      author='Tousif Islam, Scott Field, Gaurav Khanna',
      author_email='tousifislam24@gmail.com',
      #packages=['BHPTNRremnant'],
      #packages=find_packages(),
      packages=['gw_remnant', 'gw_remnant.gw_utils', 'gw_remnant.remnant_calculators'],
      license='MIT',
      include_package_data=True,
      contributors=['Tousif Islam, Scott Field, Gaurav Khanna'],
      description='Python package to extract remnant black hole properties from waveforms ',
      long_description=long_description,
      long_description_content_type='text/markdown',
      # will start new downloads if these are installed in a non-standard location
      # install_requires=["numpy","matplotlib","scipy"],
      classifiers=[
                'Intended Audience :: Other Audience',
                'Intended Audience :: Science/Research',
                'Natural Language :: English',
                'License :: OSI Approved :: MIT License',
                'Programming Language :: Python',
                'Topic :: Scientific/Engineering',
                'Topic :: Scientific/Engineering :: Mathematics',
                'Topic :: Scientific/Engineering :: Physics',
      ],
)
