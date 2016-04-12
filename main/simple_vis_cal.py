# Simple script to do basic visiblity-based calibration
import numpy as NP

def vis_cal(visdata,vismodel,max_iter=2000,threshold=0.0001):

	nchan = visdata.shape[-1]
	nant = visdata.shape[0]
	gains=NP.ones((nant,nchan),dtype=NP.complex64)

	# set up bookkeeping
	ant1 = NP.arange(nant)

	chi_history = NP.zeros((max_iter,nchan))

	tries = 0.0
	change = 100.0
	A=NP.zeros((nant**2,nant),dtype=NP.complex64) # matrix for minimization
	ind1 = NP.arange(nant**2)
	ind2 = NP.repeat(NP.arange(nant),nant)
	ind3 = NP.tile(NP.arange(nant),nant)
	for fi in xrange(nchan):
		tempgains = gains[:,fi].copy()
		tempdata = visdata[:,:,fi].reshape(-1)
		tempmodel = vismodel[:,:,fi].reshape(-1)
		while (tries < max_iter) and (change > threshold):
			chi_history[tries,fi] = NP.sum(tempdata-NP.outer(tempgains,NP.conj(tempgains))*tempmodel)
			prevgains = tempgains.copy()
			A[ind1,ind2] = tempmodel*NP.conj(prevgains[ind3])
			tempgains = NP.linalg.lstsq(A, tempdata)[0]
			change = NP.median(NP.abs(tempgains-prevgains)/NP.abs(prevgains))
			tries += 1
		if tries == max_iter:
			print 'Warning! Vis calibration failed to converge. Continuing'
		gains[:,fi] = tempgains.copy()

	return gains



