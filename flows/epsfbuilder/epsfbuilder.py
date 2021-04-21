# -*- coding: utf-8 -*-
"""
Photutils hack for EPSF building

.. codeauthor:: Simon Holmbo <simonholmbo@phys.au.dk>
"""
import time
import numpy as np
from scipy.interpolate import griddata
import photutils.psf

class EPSFBuilder(photutils.psf.EPSFBuilder):
    def _create_initial_epsf(self, stars):

        epsf = super()._create_initial_epsf(stars)
        epsf.origin = None

        X, Y = np.meshgrid(*map(np.arange, epsf.shape[::-1]))

        X = X / epsf.oversampling[0] - epsf.x_origin
        Y = Y / epsf.oversampling[1] - epsf.y_origin

        self._epsf_xy_grid = X, Y

        return epsf

    def _resample_residual(self, star, epsf):

        #max_dist = .5 / np.sqrt(np.sum(np.power(epsf.oversampling, 2)))

        #star_points = list(zip(star._xidx_centered, star._yidx_centered))
        #epsf_points = list(zip(*map(np.ravel, self._epsf_xy_grid)))

        #star_tree = cKDTree(star_points)
        #dd, ii = star_tree.query(epsf_points, distance_upper_bound=max_dist)
        #mask = np.isfinite(dd)

        #star_data = np.full_like(epsf.data, np.nan)
        #star_data.ravel()[mask] = star._data_values_normalized[ii[mask]]

        star_points = list(zip(star._xidx_centered, star._yidx_centered))
        star_data = griddata(star_points, star._data_values_normalized,
                             self._epsf_xy_grid)

        return star_data - epsf._data

    def __call__(self, *args, **kwargs):

        t0 = time.time()

        epsf, stars = super().__call__(*args, **kwargs)

        epsf.fit_info = dict(
            n_iter=len(self._epsf),
            max_iters=self.maxiters,
            time=time.time() - t0,
        )

        return epsf, stars
