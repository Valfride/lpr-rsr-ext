#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# Rayson Laroca

import os
import re
import cv2
import copy
import json
import argparse
import numpy as np

from keras.models import model_from_json

def check_format(choices, fname):
	ext = os.path.splitext(fname)[1]
	if ext not in choices:
		raise argparse.ArgumentTypeError('invalid extension format: {} (choose from {})'.format(arg, ext, choices))

	return fname

def format_command_line(string):
	idxs = [] 

	split = string.split()
	for idx, i in enumerate(split):
		if re.match(r'^[-]{1,2}[a-zA-Z]{1}', i) is not None:  
			idxs.append(idx)

	last = idxs[0]
	formated = ' '.join(split[:idxs[0]])

	for idx in idxs[1:]:
		for x in range(last, idx):
			formated += ' '

			if is_number_regex(split[x].lstrip('-')):
				formated += split[x]
			else:
				if x == last+1:
					formated += '\''

				formated += split[x]
				if x == idx - 1:
					formated += '\''
		last = idx

	return formated

def format_directory(string):
	string = format_path(string)
	if string[-1] != '/':
		string += '/'

	return string

def format_path(string):
	while '//' in string:
		string = string.replace('//', '/')

	while '/./' in string:
		string = string.replace('/./', '/')

	return string

def load_model(path):
	with open(path + '/model.json', 'r') as f:
		json = f.read()

	model = model_from_json(json)
	model.load_weights(path + '/weights.hdf5')
        
	return model

def is_number_regex(s):
	# returns 1 if is string is a number
	if re.match(r'^\d+?\.\d+?$', s) is None:
		return s.isdigit()
	return True

def padding(img, min_ratio, max_ratio, color = (0, 0, 0)):
	img_h, img_w = np.shape(img)[:2]

	border_w = 0
	border_h = 0
	ar = float(img_w)/img_h

	if ar >= min_ratio and ar <= max_ratio:
		return img, border_w, border_h

	if ar < min_ratio:
		while ar < min_ratio:
			border_w += 1
			ar = float(img_w+border_w)/(img_h+border_h)
	else:
		while ar > max_ratio:
			border_h += 1
			ar = float(img_w)/(img_h+border_h)

	border_w = border_w//2
	border_h = border_h//2

	img = cv2.copyMakeBorder(img, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value = color)
	return img, border_w, border_h

def print_dict(parameters, string = '', end='\n'):
	parameters = copy.deepcopy(parameters)
	if not isinstance(parameters, dict):
		parameters = vars(parameters)

	if len(string) > 0:
		print('\n{}:'.format(string))
	
	for key in sorted(parameters, key=str.lower):
		value = parameters[key]
		if value == '':
			value = '\'\''
			
		if isinstance(value, int) and key not in ['seed']:
			print('  {} = {:,}'.format(key, value))
		else:	
			print('  {} = {}'.format(key, value))
	print(end=end)

def print_parameters(parameters):
	print_dict(parameters, string = 'Parameters')

def save_parameters(path, parameters, save_npy = True, increment = True, zfill = 3, debug = False):
	parameters = copy.deepcopy(parameters)

	if debug:
		print('Saving the parameters...')

	filename, file_extension = os.path.splitext(path)
	if file_extension.lower() in ['.json', '.txt']:
		try:
			if increment:
				count = 1
				while os.path.exists(path):
					path = filename + '-{}'.format(str(count).zfill(zfill)) + file_extension 
					count += 1

			if file_extension == '.txt':
				with open(path, 'w') as f:
					f.writelines('parameters:\n')
					for k, v in sorted(parameters.items()):
						f.writelines('  {}: {}\n'.format(k, v))
			else:
				# json
				aux = copy.deepcopy(parameters)
				aux.pop('losses', None) # Object of type MeanSquaredError is not JSON serializable
				aux.pop('metrics', None) # Object of type CategoricalAccuracy is not JSON serializable
				with open(path, 'w') as f:
					f.write(json.dumps(aux, sort_keys=True, indent=4))

			if save_npy:
				np.save(path.replace(file_extension, '.npy'), parameters)

		except:
			raise Exception('error when saving the parameters!')
	else:
		raise Exception('unknown file format \'{}\'!'.format(file_extension))

def str2bool(value):
	if isinstance(value, bool):
		return value
	else:
		value = str(value)
		
	if value.lower() in ['yes', 'true', 'on', 't', 'y', '1']:
		return True
	elif value.lower() in ('no', 'false', 'off', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
