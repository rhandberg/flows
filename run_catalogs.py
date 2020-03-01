# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:43:21 2020

@author: au195407
"""

import argparse
import logging
from flows.catalogs import get_catalog, download_catalog

if __name__ == '__main__':
	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Run TESS Photometry pipeline on single star.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	args = parser.parse_args()

	# Set logging level:
	logging_level = logging.INFO
	if args.quiet:
		logging_level = logging.WARNING
	elif args.debug:
		logging_level = logging.DEBUG

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	console = logging.StreamHandler()
	console.setFormatter(formatter)
	logger = logging.getLogger('flows')
	logger.addHandler(console)
	logger.setLevel(logging_level)

	download_catalog(2)
	get_catalog(2)