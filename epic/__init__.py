import os as _os

__version__='0.1.0'
__description__='E-field Parallel Imaging Correlator'
__author__='Nithyanandan Thyagarajan'
__authoremail__='nithyanandan.t@gmail.com'
__maintainer__='Nithyanandan Thyagarajan'
__maintaineremail__='nithyanandan.t@gmail.com'
__url__='http://github.com/nithyanandan/epic'

with open(_os.path.dirname(_os.path.abspath(__file__))+'/githash.txt', 'r') as _githash_file:
    __githash__ = _githash_file.readline()
