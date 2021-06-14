#!/usr/bin/env python

import sys
import pickle
import numpy as np


from chimera import Chimera

class MeritCalculator:

		def __init__(self):
				self.absolutes = np.array([-0.67, 27542.28703338169, 0.2, -0.15848931924611134])
				self.relatives = np.array([np.nan, np.nan, np.nan, np.nan])
				self.chimera = Chimera(relatives=self.relatives, absolutes=self.absolutes, softness=1e-3)

		def get_observations(self, params, measurements):
				''' constructs merits and assembles observations from merits
				'''
				# params = [{'param_0': ??, 'param_1': ??, ...}, ...]
				# measurements = {'pot_fit_gibbs': [??, ??, ...], 'logd_chemaxon': [??, ??, ...], 'scscore': [??, ??, ...]}

				# construct hierarchy
				objs = np.array([
						  -np.array(measurements['peak_score']),
						   np.array(measurements['route_score']),
						   np.array(measurements['spectral_overlap']),
						  -np.array(measurements['fluo_rate'])]).T

				# compute scalarized merits
				merits, self.new_absolutes, self.thresholds, self.abs_thresh = self.chimera.scalarize(objs)
				merits = (merits - np.amin(merits)) / (np.amax(merits) - np.amin(merits))

				# construct observations
				observations = []
				for idxs, param in enumerate(params):
						#print('PARAM: ',param)
						obs = param.copy()
						obs['merit'] = merits[idxs]
						obs['peak_score'] = measurements['peak_score'][idxs]
						obs['route_score'] = measurements['route_score'][idxs]
						obs['spectral_overlap'] = measurements['spectral_overlap'][idxs]
						obs['fluo_rate'] = measurements['fluo_rate'][idxs]
						observations.append(obs)
				return observations
