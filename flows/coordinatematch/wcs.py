# -*- coding: utf-8 -*-
"""
WCS tools

.. codeauthor:: Simon Holmbo <simonholmbo@phys.au.dk>
"""
from copy import deepcopy

import numpy as np
import astropy.wcs

from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


class WCS2():
    '''Manipulate WCS solution.

    Initialize
    ----------
    wcs = WCS2(x, y, ra, dec, scale, mirror, angle)
    wcs = WCS2.from_matrix(x, y, ra, dec, matrix)
    wcs = WCS2.from_points(list(zip(x, y)), list(zip(ra, dec)))
    wcs = WCS2.from_astropy_wcs(astropy.wcs.WCS())

    ra, dec and angle should be in degrees
    scale should be in arcsec/pixel
    matrix should be the PC or CD matrix

    Examples
    --------
    Adjust x, y offset:
    wcs.x += delta_x
    wcs.y += delta_y

    Get scale and angle:
    print(wcs.scale, wcs.angle)

    Change an astropy.wcs.WCS (wcs) angle
    wcs = WCS2(wcs)(angle=new_angle).astropy_wcs

    Adjust solution with points
    wcs.adjust_with_points(list(zip(x, y)), list(zip(ra, dec)))
    '''
    def __init__(self, x, y, ra, dec, scale, mirror, angle):

        self.x, self.y = x, y
        self.ra, self.dec = ra, dec
        self.scale = scale
        self.mirror = mirror
        self.angle = angle

    @classmethod
    def from_matrix(cls, x, y, ra, dec, matrix):
        '''Initiate the class with a matrix.'''

        assert np.shape(matrix) == (2, 2), \
                'Matrix must be 2x2'

        scale, mirror, angle = cls._decompose_matrix(matrix)

        return cls(x, y, ra, dec, scale, mirror, angle)

    @classmethod
    def from_points(cls, xy, rd):
        '''Initiate the class with at least pixel + sky coordinates.'''

        assert np.shape(xy) == np.shape(rd) == (len(xy), 2) and len(xy) > 2, \
                'Arguments must be lists of at least 3 sets of coordinates'

        xy, rd = np.array(xy), np.array(rd)

        x, y, ra, dec, matrix = cls._solve_from_points(xy, rd)
        scale, mirror, angle = cls._decompose_matrix(matrix)

        return cls(x, y, ra, dec, scale, mirror, angle)

    @classmethod
    def from_astropy_wcs(cls, astropy_wcs):
        '''Initiate the class with an astropy.wcs.WCS object.'''

        assert type(astropy_wcs) is astropy.wcs.WCS, \
                'Must be astropy.wcs.WCS'

        (x, y), (ra, dec) = astropy_wcs.wcs.crpix, astropy_wcs.wcs.crval
        scale, mirror, angle = cls._decompose_matrix(
            astropy_wcs.pixel_scale_matrix)

        return cls(x, y, ra, dec, scale, mirror, angle)

    def adjust_with_points(self, xy, rd):
        '''Adjust the WCS with pixel + sky coordinates.

        If one set is given the change will be a simple offset.
        If two sets are given the offset, angle and scale will be derived.
        And if more sets are given a completely new solution will be found.
        '''

        assert np.shape(xy) == np.shape(rd) == (len(xy), 2), \
                'Arguments must be lists of sets of coordinates'

        xy, rd = np.array(xy), np.array(rd)

        self.x, self.y = xy.mean(axis=0)
        self.ra, self.dec = rd.mean(axis=0)

        A, b = xy - xy.mean(axis=0), rd - rd.mean(axis=0)
        b[:, 0] *= np.cos(np.deg2rad(rd[:, 1]))

        if len(xy) == 2:

            M = np.diag([[-1, 1][self.mirror], 1])

            def R(t):
                return np.array([[np.cos(t), -np.sin(t)],
                                 [np.sin(t), np.cos(t)]])

            def chi2(x):
                return np.power(
                    A.dot(x[1] / 60 / 60 * R(x[0]).dot(M).T) - b, 2).sum()
            self.angle, self.scale = minimize(chi2, [self.angle, self.scale]).x

        elif len(xy) > 2:

            matrix = np.linalg.lstsq(A, b, rcond=None)[0].T
            self.scale, self.mirror, self.angle = self._decompose_matrix(
                matrix)

    @property
    def matrix(self):

        scale = self.scale / 60 / 60
        mirror = np.diag([[-1, 1][self.mirror], 1])
        angle = np.deg2rad(self.angle)

        matrix = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])

        return scale * matrix @ mirror

    @property
    def astropy_wcs(self):

        wcs = astropy.wcs.WCS()
        wcs.wcs.crpix = self.x, self.y
        wcs.wcs.crval = self.ra, self.dec
        wcs.wcs.pc = self.matrix

        return wcs

    @staticmethod
    def _solve_from_points(xy, rd):

        (x, y), (ra, dec) = xy.mean(axis=0), rd.mean(axis=0)

        A, b = xy - xy.mean(axis=0), rd - rd.mean(axis=0)
        b[:, 0] *= np.cos(np.deg2rad(rd[:, 1]))

        matrix = np.linalg.lstsq(A, b, rcond=None)[0].T

        return x, y, ra, dec, matrix

    @staticmethod
    def _decompose_matrix(matrix):

        scale = np.sqrt(np.power(matrix, 2).sum() / 2) * 60 * 60

        if np.argmax(np.power(matrix[0], 2)):
            mirror = True if np.sign(matrix[0, 1]) != np.sign(
                matrix[1, 0]) else False
        else:
            mirror = True if np.sign(matrix[0, 0]) == np.sign(
                matrix[1, 1]) else False

        matrix = matrix if mirror else matrix.dot(np.diag([-1, 1]))

        matrix3d = np.eye(3)
        matrix3d[:2, :2] = matrix / (scale / 60 / 60)
        angle = Rotation.from_matrix(matrix3d).as_euler('xyz', degrees=True)[2]

        return scale, mirror, angle

    def __setattr__(self, name, value):

        if name == 'ra':

            assert 0 <= value < 360, '0 <= R.A. < 360'

        elif name == 'dec':

            assert -180 <= value <= 180, '-180 <= Dec. <= 180'

        elif name == 'scale':

            assert value > 0, 'Scale > 0'

        elif name == 'mirror':

            assert type(value) is bool, 'Mirror = True | False'

        elif name == 'angle':

            assert -180 < value <= 180, '-180 < Angle <= 180'

        super().__setattr__(name, value)

    def __call__(self, **kwargs):
        '''Make a copy with, or a copy with changes.'''

        keys = ('x', 'y', 'ra', 'dec', 'scale', 'mirror', 'angle')

        if not all(k in keys for k in kwargs):

            raise Exception('unknown argument(s)')

        obj = deepcopy(self)

        for k, v in kwargs.items():

            obj.__setattr__(k, v)

        return obj

    def __repr__(self):

        ra, dec = self.astropy_wcs.wcs_pix2world([(0, 0)], 0)[0]

        return f'WCS2(0, 0, {ra:.4f}, {dec:.4f}, {self.scale:.2f}, {self.mirror}, {self.angle:.2f})'
