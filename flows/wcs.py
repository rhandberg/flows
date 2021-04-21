#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean bad source extraction, find and correct wcs

.. codeauthor:: Emir Karamehmetoglu <emir.k@phys.au.dk>
"""

import numpy as np
import astroalign as aa
import astropy.coordinates as coords
import astropy.wcs as wcs
from astropy.stats import sigma_clip, gaussian_fwhm_to_sigma
from astropy.modeling import models, fitting
from copy import deepcopy
from bottleneck import nanmedian


class MinStarError(RuntimeError):
    pass


def force_reject_g2d(xarray,
                     yarray,
                     image,
                     get_fwhm=True,
                     rsq_min=0.5,
                     radius=10,
                     fwhm_guess=6.0,
                     fwhm_min=3.5,
                     fwhm_max=18.0):
    '''xarray, yarray, image, get_fwhm=True, rsq_min=0.5, radius=10, fwhm_guess=6.0, fwhm_min=3.5,
                    fwhm_max=18.0'''
    # Set up 2D Gaussian model for fitting to reference stars:
    g2d = models.Gaussian2D(amplitude=1.0,
                            x_mean=radius,
                            y_mean=radius,
                            x_stddev=fwhm_guess * gaussian_fwhm_to_sigma)
    g2d.amplitude.bounds = (0.1, 2.0)
    g2d.x_mean.bounds = (0.5 * radius, 1.5 * radius)
    g2d.y_mean.bounds = (0.5 * radius, 1.5 * radius)
    g2d.x_stddev.bounds = (fwhm_min * gaussian_fwhm_to_sigma,
                           fwhm_max * gaussian_fwhm_to_sigma)
    g2d.y_stddev.tied = lambda model: model.x_stddev
    g2d.theta.fixed = True

    gfitter = fitting.LevMarLSQFitter()

    # Stars reject
    N = len(xarray)
    fwhms = np.full((N, 2), np.NaN)
    xys = np.full((N, 2), np.NaN)
    rsqs = np.full(N, np.NaN)
    for i, (x, y) in enumerate(zip(xarray, yarray)):
        x = int(np.round(x))
        y = int(np.round(y))
        xmin = max(x - radius, 0)
        xmax = min(x + radius + 1, image.shape[1])
        ymin = max(y - radius, 0)
        ymax = min(y + radius + 1, image.shape[0])

        curr_star = deepcopy(image.subclean[ymin:ymax, xmin:xmax])

        edge = np.zeros_like(curr_star, dtype='bool')
        edge[(0, -1), :] = True
        edge[:, (0, -1)] = True
        curr_star -= nanmedian(curr_star[edge])
        curr_star /= np.max(curr_star)

        ypos, xpos = np.mgrid[:curr_star.shape[0], :curr_star.shape[1]]
        gfit = gfitter(g2d, x=xpos, y=ypos, z=curr_star)

        # Center
        xys[i] = np.array([gfit.x_mean + x - radius, gfit.y_mean + y - radius],
                          dtype=np.float64)
        # Calculate rsq
        sstot = ((curr_star - curr_star.mean())**2).sum()
        sserr = (gfitter.fit_info['fvec']**2).sum()
        rsqs[i] = 1. - (sserr / sstot)
        # FWHM
        fwhms[i] = gfit.x_fwhm

    masked_xys = np.ma.masked_array(xys, ~np.isfinite(xys))
    masked_rsqs = np.ma.masked_array(rsqs, ~np.isfinite(rsqs))
    mask = (masked_rsqs >= rsq_min) & (masked_rsqs < 1.0
                                       )  # Reject Rsq < rsq_min
    # changed
    #masked_xys = masked_xys[mask] # Clean extracted array.
    # to
    masked_xys.mask[~mask] = True
    # don't know if it breaks anything, but it doesn't make sence if
    # len(masked_xys) != len(masked_rsqs) FIXME
    masked_fwhms = np.ma.masked_array(fwhms, ~np.isfinite(fwhms))

    if get_fwhm: return masked_fwhms, masked_xys, mask, masked_rsqs
    return masked_xys, mask, masked_rsqs


def clean_with_rsq_and_get_fwhm(masked_fwhms,
                                masked_rsqs,
                                references,
                                min_fwhm_references=2,
                                min_references=6,
                                rsq_min=0.15):
    """
    Clean references and obtain fwhm using RSQ values.
    Args:
     masked_fwhms (np.ma.maskedarray): array of fwhms
         masked_rsqs (np.ma.maskedarray): array of rsq values
         references (astropy.table.Table): table or reference stars
         min_fwhm_references: (Default 2) min stars to get a fwhm
         min_references: (Default 6) min stars to aim for when cutting by R2
         rsq_min: (Default 0.15) min rsq value
    """
    min_references_now = min_references
    rsqvals = np.arange(rsq_min, 0.95, 0.15)[::-1]
    fwhm_found = False
    min_references_achieved = False

    # Clean based on R^2 Value
    while not min_references_achieved:
        for rsqval in rsqvals:
            mask = (masked_rsqs >= rsqval) & (masked_rsqs < 1.0)
            nreferences = np.sum(np.isfinite(masked_fwhms[mask]))
            if nreferences >= min_fwhm_references:
                _fwhms_cut_ = np.nanmean(
                    sigma_clip(masked_fwhms[mask], maxiters=100, sigma=2.0))
                #logger.info('R^2 >= ' + str(rsqval) + ': ' + str(
                #   np.sum(np.isfinite(masked_fwhms[mask]))) + ' stars w/ mean FWHM = ' + str(np.round(_fwhms_cut_, 1)))
                if not fwhm_found:
                    fwhm = _fwhms_cut_
                    fwhm_found = True
            if nreferences >= min_references_now:
                references = references[mask]
                min_references_achieved = True
                break
        if min_references_achieved: break
        min_references_now = min_references_now - 2
        if (min_references_now < 2) and fwhm_found:
            break
        elif not fwhm_found:
            raise Exception("Could not estimate FWHM")
        #logger.debug('{} {} {}'.format(min_references_now, min_fwhm_references, nreferences))

    #logger.info("FWHM: %f", fwhm)
    if np.isnan(fwhm):
        raise Exception("Could not estimate FWHM")

    # if minimum references not found, then take what we can get with even a weaker cut.
    # TODO: Is this right, or should we grab rsq_min (or even weaker?)
    min_references_now = min_references - 2
    while not min_references_achieved:
        mask = (masked_rsqs >= rsq_min) & (masked_rsqs < 1.0)
        nreferences = np.sum(np.isfinite(masked_fwhms[mask]))
        if nreferences >= min_references_now:
            references = references[mask]
            min_references_achieved = True
        rsq_min = rsq_min - 0.07
        min_references_now = min_references_now - 1

    # Check len of references as this is a destructive cleaning.
    # if len(references) == 2: logger.info('2 reference stars remaining, check WCS and image quality')
    if len(references) < 2:
        raise Exception("{} References remaining; could not clean.".format(
            len(references)))
    return fwhm, references


def mkposxy(posx, posy):
    '''Make 2D np array for astroalign'''
    img_posxy = np.array([[x, y] for x, y in zip(posx, posy)], dtype="float64")
    return img_posxy


def try_transform(source, target, pixeltol=2, nnearest=5, max_stars=50):
    aa.NUM_NEAREST_NEIGHBORS = nnearest
    aa.PIXEL_TOL = pixeltol
    transform, (sourcestars,
                targetstars) = aa.find_transform(source,
                                                 target,
                                                 max_control_points=max_stars)
    return sourcestars, targetstars


def try_astroalign(source, target, pixeltol=2, nnearest=5, max_stars_n=50):
    # Get indexes of matched stars
    success = False
    try:
        source_stars, target_stars = try_transform(source,
                                                   target,
                                                   pixeltol=pixeltol,
                                                   nnearest=nnearest,
                                                   max_stars=max_stars_n)
        source_ind = np.argwhere(np.in1d(source, source_stars)[::2]).flatten()
        target_ind = np.argwhere(np.in1d(target, target_stars)[::2]).flatten()
        success = True
    except aa.MaxIterError:
        source_ind, target_ind = 'None', 'None'
    return source_ind, target_ind, success


def min_to_max_astroalign(source,
                          target,
                          fwhm=5,
                          fwhm_min=1,
                          fwhm_max=4,
                          knn_min=5,
                          knn_max=20,
                          max_stars=100,
                          min_matches=3):
    '''Try to find matches using astroalign asterisms by stepping through some parameters.'''
    # Set max_control_points par based on number of stars and max_stars.
    nstars = max(len(source), len(source))
    if max_stars >= nstars: max_stars_list = 'None'
    else:
        if max_stars > 60: max_stars_list = (max_stars, 50, 4, 3)
        else: max_stars_list = (max_stars, 6, 4, 3)

    # Create max_stars step-through list if not given
    if max_stars_list == 'None':
        if nstars > 6:
            max_stars_list = (nstars, 5, 3)
        elif nstars > 3:
            max_stars_list = (nstars, 3)

    pixeltols = np.linspace(int(fwhm * fwhm_min),
                            int(fwhm * fwhm_max),
                            4,
                            dtype=int)
    nearest_neighbors = np.linspace(knn_min,
                                    min(knn_max, nstars),
                                    4,
                                    dtype=int)

    for max_stars_n in max_stars_list:
        for pixeltol in pixeltols:
            for nnearest in nearest_neighbors:
                source_ind, target_ind, success = try_astroalign(
                    source,
                    target,
                    pixeltol=pixeltol,
                    nnearest=nnearest,
                    max_stars_n=max_stars_n)
                if success:
                    if len(source_ind) >= min_matches:
                        return source_ind, target_ind, success
                    else:
                        success = False
    return 'None', 'None', success


def kdtree(source, target, fwhm=5, fwhm_max=4, min_matches=3):
    '''Use KDTree to get nearest neighbor matches within fwhm_max*fwhm distance'''

    # Use KDTree to rapidly efficiently query nearest neighbors
    from scipy.spatial import KDTree
    tt = KDTree(target)
    st = KDTree(source)
    matches_list = st.query_ball_tree(tt, r=fwhm * fwhm_max)

    #indx = []
    targets = []
    sources = []
    for j, (sstar, match) in enumerate(zip(source, matches_list)):
        if np.array(target[match]).size != 0:
            targets.append(match[0])
            sources.append(j)
    sources = np.array(sources, dtype=int)
    targets = np.array(targets, dtype=int)
    # Return indexes of matches
    return sources, targets, len(sources) >= min_matches


def get_new_wcs(extracted_ind,
                extracted_stars,
                clean_references,
                ref_ind,
                obstime,
                rakey='ra_obs',
                deckey='decl_obs'):
    targets = (extracted_stars[extracted_ind][:, 0],
               extracted_stars[extracted_ind][:, 1])
    c = coords.SkyCoord(clean_references[rakey][ref_ind],
                        clean_references[deckey][ref_ind],
                        obstime=obstime)
    return wcs.utils.fit_wcs_from_points(targets, c)


def get_clean_references(references,
                         masked_rsqs,
                         min_references_ideal=6,
                         min_references_abs=3,
                         rsq_min=0.15,
                         rsq_ideal=0.5,
                         keep_max=100,
                         rescue_bad: bool = True):

    # Greedy first try
    mask = (masked_rsqs >= rsq_ideal) & (masked_rsqs < 1.0)
    if np.sum(np.isfinite(masked_rsqs[mask])) >= min_references_ideal:
        if len(references[mask]) <= keep_max:
            return references[mask]
        elif len(references[mask]) >= keep_max:
            import pandas as pd  # @TODO: Convert to pure numpy implementation
            df = pd.DataFrame(masked_rsqs, columns=['rsq'])
            masked_rsqs.mask = ~mask
            nmasked_rsqs = df.sort_values('rsq',
                                          ascending=False).dropna().index._data
            return references[nmasked_rsqs[:keep_max]]

    # Desperate second try
    mask = (masked_rsqs >= rsq_min) & (masked_rsqs < 1.0)
    masked_rsqs.mask = ~mask

    # Switching to pandas for easier selection
    import pandas as pd  # @TODO: Convert to pure numpy implementation
    df = pd.DataFrame(masked_rsqs, columns=['rsq'])
    nmasked_rsqs = deepcopy(
        df.sort_values('rsq', ascending=False).dropna().index._data)
    nmasked_rsqs = nmasked_rsqs[:min(min_references_ideal, len(nmasked_rsqs))]
    if len(nmasked_rsqs) >= min_references_abs:
        return references[nmasked_rsqs]
    if not rescue_bad:
        raise MinStarError(
            'Less than {} clean stars and rescue_bad = False'.format(
                min_references_abs))

    # Extremely desperate last ditch attempt i.e. "rescue bad"
    elif rescue_bad:
        mask = (masked_rsqs >= 0.02) & (masked_rsqs < 1.0)
        masked_rsqs.mask = ~mask

        # Switch to pandas
        df = pd.DataFrame(masked_rsqs, columns=['rsq'])
        nmasked_rsqs = df.sort_values('rsq',
                                      ascending=False).dropna().index._data
        nmasked_rsqs = nmasked_rsqs[:min(min_references_ideal, len(nmasked_rsqs
                                                                   ))]
        if len(nmasked_rsqs) < 2:
            raise MinStarError('Less than 2 clean stars.')
        return references[nmasked_rsqs]  # Return if len >= 2
    # Checks whether sensible input arrays and parameters were provided
    raise ValueError(
        'input parameters were wrong, you should not reach here. Check that rescue_bad is True or False.'
    )
