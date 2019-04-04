import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

LONG_DESCRIPTION = '''
Pywebplot is a module for python to draw interactive maps and plots by javascript.
It is mainly based on mapbox_gl.js and d3.js.
Docs for mapbox_gl.js: https://docs.mapbox.com/mapbox-gl-js/style-spec/
Docs for d3.js: https://d3js.org/
'''

if os.environ.get('READTHEDOCS', False) == 'True':
    INSTALL_REQUIRES = []
else:
    INSTALL_REQUIRES = ['webbrowser', 'geopandas', 'shapely', 'numpy']

setup(name='pywebplot',
      description='Web plot for python.',
      license='GPL',
      version='0.1.0',
      author='LaZzy',
      author_email='zeusdream7@gmail.com',
      url='',
      long_description=LONG_DESCRIPTION,
      install_requires=INSTALL_REQUIRES)
