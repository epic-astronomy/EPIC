import setuptools, re, glob, os
from setuptools import setup, find_packages
from subprocess import Popen, PIPE

githash = 'unknown'
if os.path.isdir(os.path.dirname(os.path.abspath(__file__))+'/.git'):
    try:
        gitproc = Popen(['git', 'rev-parse', 'HEAD'], stdout = PIPE)
        githash = gitproc.communicate()[0]
        if gitproc.returncode != 0:
            print "unable to run git, assuming githash to be unknown"
            githash = 'unknown'
    except EnvironmentError:
        print "unable to run git, assuming githash to be unknown"
githash = githash.replace('\n', '')

with open(os.path.dirname(os.path.abspath(__file__))+'/epic/githash.txt', 'w+') as githash_file:
    githash_file.write(githash)

metafile = open(os.path.dirname(os.path.abspath(__file__))+'/epic/__init__.py').read()
metadata = dict(re.findall("__([a-z]+)__\s*=\s*'([^']+)'", metafile))

setup(name='EPIC',
    version=metadata['version'],
    description=metadata['description'],
    long_description=open("README.rst").read(),
    url=metadata['url'],
    author=metadata['author'],
    author_email=metadata['authoremail'],
    license='MIT',
    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 2.7',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Astronomy',
                 'Topic :: Utilities'],
    packages=find_packages(),
    package_data={'epic': ['*.txt', 'examples/ioparms/*.yaml']},
    include_package_data=True,
    scripts=glob.glob('scripts/*.py') + ['LWA/LWA_bifrost.py'],
    install_requires=['astropy>=1.0', 'astroutils>=0.1.0', 'ipdb>=0.6.1',
                      'matplotlib>=1.4.3', 'numpy>=1.8.1', 'progressbar>=2.3',
                      'pyyaml>=3.11', 'scipy>=0.15.1', 'h5py>=2.7.0'],
    setup_requires=['astropy>=1.0', 'astroutils>=0.1.0', 'ipdb>=0.6.1',
                    'matplotlib>=1.4.3', 'numpy>=1.8.1', 'progressbar>=2.3',
                    'pytest-runner', 'pyyaml>=3.11', 'scipy>=0.15.1',
                    'h5py>=2.7.0'],
    tests_require=['pytest'],
    zip_safe=False)
