import numpy as NP
import os
import argparse
import yaml
from lsl.common.stations import lwa1, lwasv
import epic
from epic import data_interface as DI
import ipdb as PDB

epic_path = epic.__path__[0]+'/'

if __name__ == '__main__':

    ## Parse input arguments
    
    parser = argparse.ArgumentParser(description='Program to simulate interferometer array data')
    
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-i', '--infile', dest='infile', default=epic_path+'examples/ioparms/LWASV_TBN_input_file_parameters.yaml', type=str, required=False, help='File specifying input parameters')
    
    args = vars(parser.parse_args())

    with open(args['infile'], 'r') as parms_file:
        parms = yaml.safe_load(parms_file)

    ioparms = parms['dirstruct']
    indir = ioparms['indir']
    infile = indir + ioparms['infile']
    outdir = ioparms['outdir']
    outfile = outdir + ioparms['outfile']

    instrumentinfo = parms['instrumentinfo']
    station_name = instrumentinfo['station']

    if station_name.lower() not in ['lwa1', 'lwasv']:
        raise valueError('LWA station not recognized')

    if station_name.lower() == 'lwa1':
        station = lwa1
    else:
        station = lwasv

    
