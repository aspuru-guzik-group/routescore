#!/usr/bin/env python

import os
import copy
import json
import pickle
import numpy as np

#=================================================================

class CategoryWriter(object):

	param_names = ['frag_a', 'frag_b', 'frag_c']

	desc_names  = {
			'frag_a':   [],
			'frag_b':   [],
			'frag_c':   [],
		}


	def __init__(self):
		self.frag_a = json.loads(open('frag_a.json', 'r').read())
		self.frag_b = json.loads(open('frag_b.json', 'r').read())
		self.frag_c = json.loads(open('frag_c.json', 'r').read())
		self.opts     = {'frag_a': self.frag_a,
						 'frag_b': self.frag_b,
						 'frag_c': self.frag_c,
					}


	def write_categories(self, home_dir, with_descriptors = True):
		for param_name in self.param_names:
			opt_list = []
			for opt_name, opt_desc_dict in self.opts[param_name].items():
				opt_dict = {'name': opt_name}
				if with_descriptors:
					opt_dict['descriptors'] = np.array([float(opt_desc_dict[desc_name]) for desc_name in self.desc_names[param_name]])
				opt_list.append(copy.deepcopy(opt_dict))

			# create cat details dir if necessary
			dir_name = '%s/CatDetails' % home_dir
			if not os.path.isdir(dir_name): os.mkdir(dir_name)

			cat_details_file = '%s/cat_details_%s.pkl' % (dir_name, param_name)
			with open(cat_details_file, 'wb') as content:
				pickle.dump(opt_list, content)

#=================================================================

if __name__ == '__main__':

	cat_writer = CatWriter()
	cat_writer.write_cats('./')
