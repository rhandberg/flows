#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot photometry for target, loaded from local photometry working directory.

.. codeauthor:: Emir K <emir.k@phys.au.dk>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import argparse
import os
import glob
import numpy as np
from astropy.table import Table
from flows.plots import plt, plots_interactive
from flows import api, load_config
import mplcursors
import seaborn as sns

#--------------------------------------------------------------------------------------------------
def main():
	# All available filters:
	all_filters = list(api.get_filters().keys())

	# Parser:
	parser = argparse.ArgumentParser(description='Plot photometry for target')
	parser.add_argument('--target', '-t', type=str, required=True, help="""Target identifier:
		Can be either the SN name (e.g. 2019yvr) or the Flows target ID.""")
	parser.add_argument('--fileid', '-i', type=int, nargs='*', default=None,
		help='Specific file ids within target separated by spaces: -i <ID> <ID> <ID>')
	parser.add_argument('--filters', '-f', type=str, nargs='*', default=None,
		help='List of space delimited filters. If not provided will use all. Choose between {}'.format(all_filters))
	parser.add_argument('--offset', '-jd', type=float, default=2458800.0)
	parser.add_argument('--subonly', help='Only show template subtracted data points.', action='store_true')
	parser.add_argument('--hidpi', help='double DPI fo 4k resolution', action='store_true')
	args = parser.parse_args()

	# To use when only plotting some filters
	usefilts = args.filters
	if usefilts is not None:
		usefilts = set(args.filters)

	# To use when only plotting some fileids
	# Parse input fileids:
	if args.fileid is not None:
		# Plot the specified fileid:
		fileids = args.fileid
	else:
		fileids = []
	if len(fileids) > 1:
		raise NotImplementedError("This has not been implemented yet")

	offset = args.offset

	# Get the name of the target:
	snname = args.target
	if snname.isdigit():
		datafiles = api.get_datafiles(int(snname), filt='all')
		snname = api.get_datafile(datafiles[0])['target_name']

	# Change to directory, raise if it does not exist
	config = load_config()
	workdir_root = config.get('photometry', 'output', fallback='.')
	sndir = os.path.join(workdir_root, snname)
	if not os.path.isdir(sndir):
		print('No such directory as',sndir)
		return

	# Get list of photometry files
	phot_files = glob.iglob(os.path.join(sndir, '*', 'photometry.ecsv'))

	# Load all data into astropy table
	tablerows = []
	for file in phot_files:
		# Load photometry file into Table:
		AT = Table.read(file, format='ascii.ecsv')

		# Pull out meta-data:
		fileid = AT.meta['fileid']
		filt = AT.meta['photfilter']
		jd = AT.meta['obstime-bmjd'] + 2400000.5

		# get phot of diff image
		AT.add_index('starid')
		if -1 in AT['starid']:
			mag, mag_err = AT.loc[-1]['mag'], AT.loc[-1]['mag_error']
			sub = True
		elif 0 in AT['starid']:
			print('No subtraction found for:',file,'in filter',filt)
			mag,mag_err = AT.loc[0]['mag'],AT.loc[0]['mag_error']
			sub = False
		else:
			print('No object phot found, skipping: \n',file)
			continue

		tablerows.append((jd, mag, mag_err, filt, sub, fileid))

	phot = Table(
		rows=tablerows,
		names=['jd','mag','mag_err','filter','sub','fileid'],
		dtype=['float64','float64','float64','S64','bool','int64'])

	# Create list of filters to plot:
	filters = list(np.unique(phot['filter']))
	if usefilts:
		filters = set(filters).intersection(usefilts)

	# Split photometry table
	shifts = dict(zip(filters, np.zeros(len(filters))))

	# Create the plot:
	plots_interactive()
	sns.set(style='ticks')
	dpi_mult = 1 if not args.subonly else 2
	fig, ax = plt.subplots(figsize=(6.4,4), dpi=130*dpi_mult)
	fig.subplots_adjust(top=0.95, left=0.1, bottom=0.1, right=0.97)

	cps = sns.color_palette()
	colors = dict(zip(filters,(cps[2],cps[3],cps[0],cps[-1],cps[1])))

	if args.subonly:
		for filt in filters:
			lc = phot[(phot['filter'] == filt) & phot['sub']]
			ax.errorbar(lc['jd'] - offset, lc['mag'] + shifts[filt], lc['mag_err'],
				marker='s', linestyle='None', label=filt, color=colors[filt])

	else:
		for filt in filters:
			lc = phot[phot['filter'] == filt]
			ax.errorbar(lc['jd'] - offset, lc['mag'] + shifts[filt], lc['mag_err'],
				marker='s', linestyle='None', label=filt, color=colors[filt])

	ax.invert_yaxis()
	ax.legend()
	ax.set_xlabel('JD - ' + str(offset), fontsize=16)
	ax.set_ylabel('App. Mag', fontsize=16)
	ax.set_title(snname)

	# Make the points interactive:
	def annotate(sel):
		lc = phot[phot['filter'] == str(sel.artist.get_label())]
		point = lc[sel.target.index]
		point = dict(zip(point.colnames, point)) # Convert table row to dict
		return sel.annotation.set_text("Fileid: {fileid:d}\nJD: {jd:.3f}\nMag: {mag:.2f}$\\pm${mag_err:.2f}".format(**point))

	mplcursors.cursor(ax).connect("add", annotate)
	plt.show(block=True)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	main()
