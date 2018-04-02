EPIC (E-field Parallel Imaging Correlator)
==========================================

A modular and generic direct imaging correlator for radio interferometer arrays.


Installation
============
Note that currently this package only supports Python 2.6+, and not Python 3. 

Using Anaconda
--------------
If using the Anaconda python distribution, many of the packages may be installed using ``conda``.

It is best to first create a new env:

``conda create -n YOURENV python=2.7 anaconda``

Activate this environment:

``source activate EPIC``

Then install conda packages:

``conda install progressbar psutil pyyaml h5py``

You also need ``AstroUtils``:

``pip install git+https://github.com/nithyanandan/AstroUtils.git``

which will install a list of dependencies.

Finally, either install EPIC directly:

``pip install git+https://github.com/nithyanandan/EPIC.git``

or clone it into top-level directory called ``EPIC`` and install from this
directory:

``mkdir EPIC``

``cd EPIC``

``pip install .``.


Basic Usage
===========

``run_EPIC.py -i /path/to/imagingparms.yaml``

where ``/path/to/imagingparms.yaml`` is a YAML file specifying all imaging
and processing parameters. See example in

``epic/examples/imagingparms/EPIC_parms.yaml``
