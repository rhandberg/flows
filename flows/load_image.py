#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flows photometry code.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import warnings
import astropy.units as u
import astropy.coordinates as coords
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning
from . import api

#--------------------------------------------------------------------------------------------------
def edge_mask(img, value=0):
	"""
	Create boolean mask of given value near edge of image.

	Parameters:
		img (ndarray): Image of
		value (float): Value to detect near edge. Default=0.

	Returns:
		ndarray: Pixel mask with given values on the edge of image.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	mask1 = (img == value)
	mask = np.zeros_like(img, dtype='bool')

	# Mask entire rows and columns which are only the value:
	mask[np.all(mask1, axis=1), :] = True
	mask[:, np.all(mask1, axis=0)] = True

	# Detect "uneven" edges column-wise in image:
	a = np.argmin(mask1, axis=0)
	b = np.argmin(np.flipud(mask1), axis=0)
	for col in range(img.shape[1]):
		if mask1[0, col]:
			mask[:a[col], col] = True
		if mask1[-1, col]:
			mask[-b[col]:, col] = True

	# Detect "uneven" edges row-wise in image:
	a = np.argmin(mask1, axis=1)
	b = np.argmin(np.fliplr(mask1), axis=1)
	for row in range(img.shape[0]):
		if mask1[row, 0]:
			mask[row, :a[row]] = True
		if mask1[row, -1]:
			mask[row, -b[row]:] = True

	return mask

#--------------------------------------------------------------------------------------------------
def load_image(FILENAME):
	"""
	Load FITS image.

	Parameters:
		FILENAME (string): Path to FITS file to be loaded.

	Returns:
		object: Image constainer.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	# Get image and WCS, find stars, remove galaxies
	image = type('image', (object,), dict()) # image container

	# get image and wcs solution
	with fits.open(FILENAME, mode='readonly') as hdul:

		hdr = hdul[0].header
		origin = hdr.get('ORIGIN')
		telescope = hdr.get('TELESCOP')
		instrument = hdr.get('INSTRUME')

		image.image = np.asarray(hdul[0].data, dtype='float64')
		image.shape = image.image.shape

		image.header = hdr

		if origin == 'LCOGT':
			image.mask = np.asarray(hdul['BPM'].data, dtype='bool')
		else:
			image.mask = np.zeros_like(image.image, dtype='bool')

		image.mask |= ~np.isfinite(image.image)

		# World Coordinate System:
		with warnings.catch_warnings():
			warnings.simplefilter('ignore', category=FITSFixedWarning)
			image.wcs = WCS(hdr)

		# Specific headers everyone can agree on:
		image.exptime = float(hdr['EXPTIME']) # * u.second
		image.peakmax = None # Maximum value above which data is not to be trusted

		# Timestamp:
		if origin == 'LCOGT':
			sites = api.sites.get_all_sites()
			site_keywords = {s['site_keyword']: s for s in sites}
			image.site = site_keywords.get(hdr['SITE'], None)

			observatory = coords.EarthLocation.from_geodetic(lat=hdr['LATITUDE'], lon=hdr['LONGITUD'], height=hdr['HEIGHT'])
			image.obstime = Time(hdr['DATE-OBS'], format='isot', scale='utc', location=observatory)
			image.obstime += 0.5*image.exptime * u.second # Make time centre of exposure

			image.photfilter = hdr['FILTER']

			# Get non-linear limit
			# TODO: Use actual or some fraction of the non-linearity limit
			#image.peakmax = hdr.get('MAXLIN') # Presumed non-linearity limit from header
			image.peakmax = 60000 # From experience, this one is better.

		elif telescope == 'NOT' and instrument in ('ALFOSC FASU', 'ALFOSC_FASU') and hdr.get('OBS_MODE', '').lower() == 'imaging':
			image.site = api.get_site(5) # Hard-coded the siteid for NOT
			image.obstime = Time(hdr['DATE-AVG'], format='isot', scale='utc', location=image.site['EarthLocation'])

			# Sometimes data from NOT does not have the FILTER keyword,
			# in which case we have to try to figure out which filter
			# was used based on some of the other headers:
			if 'FILTER' in hdr:
				image.photfilter = {
					'B Bes': 'B',
					'V Bes': 'V',
					'R Bes': 'R',
					'g SDSS': 'gp',
					'r SDSS': 'rp',
					'i SDSS': 'ip',
					'i int': 'ip', # Interference filter
					'u SDSS': 'up',
					'z SDSS': 'zp'
				}.get(hdr['FILTER'].replace('_', ' '), hdr['FILTER'])
			else:
				filters_used = []
				for check_headers in ('ALFLTNM', 'FAFLTNM', 'FBFLTNM'):
					if hdr.get(check_headers) and hdr.get(check_headers).strip().lower() != 'open':
						filters_used.append(hdr.get(check_headers).strip())
				if len(filters_used) == 1:
					image.photfilter = {
						'V_Bes 530_80': 'V',
						'R_Bes 650_130': 'R',
						"g'_SDSS 480_145": 'gp',
						"r'_SDSS 618_148": 'rp',
						"i'_SDSS 771_171": 'ip',
						'i_int 797_157': 'ip', # Interference filter
						"z'_SDSS 832_LP": 'zp'
					}.get(filters_used[0].replace('  ', ' '), filters_used[0])
				else:
					raise Exception("Could not determine filter used.")

			# Get non-linear limit
			# Obtained from http://www.not.iac.es/instruments/detectors/CCD14/LED-linearity/20181026-200-1x1.pdf
			# TODO: grab these from a table for all detector setups of ALFOSC
			image.peakmax = 80000 # For ALFOSC D, 1x1, 200; the standard for SNe.

		elif telescope == 'NOT' and instrument == 'NOTCAM' and hdr.get('OBS_MODE', '').lower() == 'imaging':
			image.site = api.get_site(5) # Hard-coded the siteid for NOT
			image.obstime = Time(hdr['DATE-AVG'], format='isot', scale='utc', location=image.site['EarthLocation'])

			# Does NOTCAM data sometimes contain a FILTER header?
			# if not we have to try to figure out which filter
			# was used based on some of the other headers:
			if 'FILTER' in hdr:
				raise Exception("NOTCAM: Filter keyword defined")
			filters_used = []
			for check_headers in ('NCFLTNM1', 'NCFLTNM2'):
				if hdr.get(check_headers) and hdr.get(check_headers).strip().lower() != 'open':
					filters_used.append(hdr.get(check_headers).strip())
			if len(filters_used) == 1:
				image.photfilter = {}.get(filters_used[0], filters_used[0])
			else:
				raise Exception("Could not determine filter used.")

			# Mask out "halo" of pixels with zero value along edge of image:
			image.mask |= edge_mask(image.image, value=0)

		elif hdr.get('FPA.TELESCOPE') == 'PS1' and hdr.get('FPA.INSTRUMENT') == 'GPC1':
			image.site = api.get_site(6) # Hard-coded the siteid for Pan-STARRS1
			image.obstime = Time(hdr['MJD-OBS'], format='mjd', scale='utc', location=image.site['EarthLocation'])

			image.photfilter = {
				'g.00000': 'gp',
				'r.00000': 'rp',
				'i.00000': 'ip',
				'z.00000': 'zp'
			}.get(hdr['FPA.FILTER'], hdr['FPA.FILTER'])

		elif telescope == 'Liverpool Telescope':
			# Liverpool telescope
			image.site = api.get_site(8) # Hard-coded the siteid for Liverpool Telescope
			image.obstime = Time(hdr['DATE-OBS'], format='isot', scale='utc', location=image.site['EarthLocation'])
			image.obstime += 0.5*image.exptime * u.second # Make time centre of exposure
			image.photfilter = {
				'Bessel-B': 'B',
				'Bessell-B': 'B',
				'Bessel-V': 'V',
				'Bessell-V': 'V',
				'SDSS-U': 'up',
				'SDSS-G': 'gp',
				'SDSS-R': 'rp',
				'SDSS-I': 'ip',
				'SDSS-Z': 'zp'
			}.get(hdr['FILTER1'], hdr['FILTER1'])

		elif telescope == 'CA 3.5m' and instrument == 'Omega2000':
			# Calar Alto 3.5m (Omege2000)
			image.site = api.get_site(9) # Hard-coded the siteid for Calar Alto 3.5m
			image.obstime = Time(hdr['MJD-OBS'], format='mjd', scale='utc', location=image.site['EarthLocation'])
			image.obstime += 0.5*image.exptime * u.second # Make time centre of exposure
			image.photfilter = hdr['FILTER']

		elif telescope == 'SWO' and hdr.get('SITENAME') == 'LCO':
			image.site = api.get_site(10) # Hard-coded the siteid for Swope, Las Campanas Observatory
			image.obstime = Time(hdr['JD'], format='jd', scale='utc', location=image.site['EarthLocation'])
			image.photfilter = {
				'u': 'up',
				'g': 'gp',
				'r': 'rp',
				'i': 'ip',
			}.get(hdr['FILTER'], hdr['FILTER'])

		elif telescope == 'DUP' and hdr.get('SITENAME') == 'LCO' and instrument == 'Direct/SITe2K-1':
			image.site = api.get_site(14) # Hard-coded the siteid for Du Pont, Las Campanas Observatory
			image.obstime = Time(hdr['JD'], format='jd', scale='utc', location=image.site['EarthLocation'])
			image.photfilter = {
				'u': 'up',
				'g': 'gp',
				'r': 'rp',
				'i': 'ip',
			}.get(hdr['FILTER'], hdr['FILTER'])

		elif telescope == 'DUP' and instrument == 'RetroCam':
			image.site = api.get_site(16) # Hard-coded the siteid for Du Pont, Las Campanas Observatory
			image.obstime = Time(hdr['JD'], format='jd', scale='utc', location=image.site['EarthLocation'])
			image.photfilter = {
				'Yc': 'Y',
				'Hc': 'H',
				'Jo': 'J',
			}.get(hdr['FILTER'], hdr['FILTER'])

		elif telescope == 'Baade' and hdr.get('SITENAME') == 'LCO' and instrument == 'FourStar':
			image.site = api.get_site(11) # Hard-coded the siteid for Swope, Las Campanas Observatory
			image.obstime = Time(hdr['JD'], format='jd', scale='utc', location=image.site['EarthLocation'])
			image.photfilter = {
				'Ks': 'K',
				'J1': 'Y',
			}.get(hdr['FILTER'], hdr['FILTER'])
			image.exptime *= int(hdr['NCOMBINE']) # EXPTIME is only for a single exposure

		elif instrument == 'SOFI' and telescope in ('ESO-NTT', 'other') and (origin == 'ESO' or origin.startswith('NOAO-IRAF')):
			image.site = api.get_site(12) # Hard-coded the siteid for SOFT, ESO NTT
			if 'TMID' in hdr:
				image.obstime = Time(hdr['TMID'], format='mjd', scale='utc', location=image.site['EarthLocation'])
			else:
				image.obstime = Time(hdr['MJD-OBS'], format='mjd', scale='utc', location=image.site['EarthLocation'])
				image.obstime += 0.5*image.exptime * u.second # Make time centre of exposure

			# Photometric filter:
			photfilter_translate = {
				'Ks': 'K'
			}
			if 'FILTER' in hdr:
				image.photfilter = photfilter_translate.get(hdr['FILTER'], hdr['FILTER'])
			else:
				filters_used = []
				for check_headers in ('ESO INS FILT1 ID', 'ESO INS FILT2 ID'):
					if hdr.get(check_headers) and hdr.get(check_headers).strip().lower() != 'open':
						filters_used.append(hdr.get(check_headers).strip())
				if len(filters_used) == 1:
					image.photfilter = photfilter_translate.get(filters_used[0], filters_used[0])
				else:
					raise Exception("Could not determine filter used.")

			# Mask out "halo" of pixels with zero value along edge of image:
			image.mask |= edge_mask(image.image, value=0)

		elif telescope == 'ESO-NTT' and instrument == 'EFOSC' and (origin == 'ESO' or origin.startswith('NOAO-IRAF')):
			image.site = api.get_site(15) # Hard-coded the siteid for EFOSC, ESO NTT
			image.obstime = Time(hdr['DATE-OBS'], format='isot', scale='utc', location=image.site['EarthLocation'])
			image.obstime += 0.5*image.exptime * u.second # Make time centre of exposure
			image.photfilter = {
				'g782': 'gp',
				'r784': 'rp',
				'i705': 'ip',
				'B639': 'B',
				'V641': 'V'
			}.get(hdr['FILTER'], hdr['FILTER'])

		elif telescope == 'SAI-2.5' and instrument == 'ASTRONIRCAM':
			image.site = api.get_site(13) # Hard-coded the siteid for Caucasus Mountain Observatory
			image.obstime = Time(hdr['MJD-AVG'], format='mjd', scale='utc', location=image.site['EarthLocation'])
			image.photfilter = {
				'H_Open': 'H',
				'K_Open': 'K',
			}.get(hdr['FILTER'], hdr['FILTER'])
			image.exptime = float(hdr['FULL_EXP'])

		elif instrument == 'OMEGACAM' and (origin == 'ESO' or origin.startswith('NOAO-IRAF')):
			image.site = api.get_site(18) # Hard-coded the siteid for ESO VLT Survey telescope
			image.obstime = Time(hdr['MJD-OBS'], format='mjd', scale='utc', location=image.site['EarthLocation'])
			image.obstime += 0.5*image.exptime * u.second # Make time centre of exposure
			image.photfilter = {
				'i_SDSS': 'ip'
			}.get(hdr['ESO INS FILT1 NAME'], hdr['ESO INS FILT1 NAME'])

		elif instrument == 'ANDICAM-CCD' and hdr.get('OBSERVAT') == 'CTIO':
			image.site = api.get_site(20) # Hard-coded the siteid for ANDICAM at Cerro Tololo Interamerican Observatory (CTIO)
			image.obstime = Time(hdr['JD'], format='jd', scale='utc', location=image.site['EarthLocation'])
			image.obstime += 0.5*image.exptime * u.second # Make time centre of exposure
			image.photfilter = hdr['CCDFLTID']

		else:
			raise Exception("Could not determine origin of image")

	# Create masked version of image:
	image.image[image.mask] = np.NaN
	image.clean = np.ma.masked_array(image.image, image.mask)

	return image
