# -*- coding: utf-8 -*-
"""
Match two sets of coordinates

.. codeauthor:: Simon Holmbo <simonholmbo@phys.au.dk>
"""
import time

from itertools import count, islice, chain, product, zip_longest

import numpy as np

from astropy.coordinates.angle_utilities import angular_separation
from scipy.spatial import cKDTree as KDTree
from networkx import Graph, connected_components

from .wcs import WCS2


class CoordinateMatch():
    def __init__(self,
                 xy,
                 rd,
                 xy_order=None,
                 rd_order=None,
                 xy_nmax=None,
                 rd_nmax=None,
                 n_triangle_packages=10,
                 triangle_package_size=10000,
                 maximum_angle_distance=0.001,
                 distance_factor=1):

        self.xy, self.rd = np.array(xy), np.array(rd)

        self._xy = xy - np.mean(xy, axis=0)
        self._rd = rd - np.mean(rd, axis=0)
        self._rd[:, 0] *= np.cos(np.deg2rad(self.rd[:, 1]))

        xy_n, rd_n = min(xy_nmax, len(xy)), min(rd_nmax, len(rd))

        self.i_xy = xy_order[:xy_n] if xy_order is not None else np.arange(
            xy_n)
        self.i_rd = rd_order[:rd_n] if rd_order is not None else np.arange(
            rd_n)

        self.n_triangle_packages = n_triangle_packages
        self.triangle_package_size = triangle_package_size

        self.maximum_angle_distance = maximum_angle_distance
        self.distance_factor = distance_factor

        self.triangle_package_generator = self._sorted_triangle_packages()

        self.i_xy_triangles = list()
        self.i_rd_triangles = list()
        self.parameters = None
        self.neighbours = Graph()

        self.normalizations = type(
            'Normalizations', (object, ),
            dict(ra=0.0001, dec=0.0001, scale=0.002, angle=0.002))

        self.bounds = type(
            'Bounds', (object, ),
            dict(xy=self.xy.mean(axis=0),
                 rd=None,
                 radius=None,
                 scale=None,
                 angle=None))

    def set_normalizations(self, ra=None, dec=None, scale=None, angle=None):
        '''Set normalization factors in the (ra, dec, scale, angle) space.

        Defaults are:
            ra = 0.0001 degrees
            dec = 0.0001 degrees
            scale = 0.002 log(arcsec/pixel)
            angle = 0.002 radians
        '''

        if self.parameters is not None:

            raise Exception(
                'can\'t change normalization after matching is started')

        assert ra is None or 0 < ra
        assert dec is None or 0 < dec
        assert scale is None or 0 < scale
        assert angle is None or 0 < angle

        self.normalizations.ra = ra if ra is not None else self.normalizations.ra
        self.normalizations.dec = dec if dec is not None else self.normalizations.dec
        self.normalizations.scale = scale if scale is not None else self.normalizations.scale
        self.normalizations.angle = angle if ra is not None else self.normalizations.angle

    def set_bounds(self,
                   x=None,
                   y=None,
                   ra=None,
                   dec=None,
                   radius=None,
                   scale=None,
                   angle=None):
        '''Set bounds for what are valid results.

        Set x, y, ra, dec and radius to specify that the x, y coordinates must be no
        further that the radius [degrees] away from the ra, dec coordinates.
        Set upper and lower bounds on the scale [log(arcsec/pixel)] and/or the angle
        [radians] if those are known, possibly from previous observations with the
        same system.
        '''

        if self.parameters is not None:

            raise Exception('can\'t change bounds after matching is started')

        if [x, y, ra, dec, radius].count(None) == 5:

            assert 0 <= ra < 360
            assert -180 <= dec <= 180
            assert 0 < radius

            self.bounds.xy = x, y
            self.bounds.rd = ra, dec
            self.bounds.radius = radius

        elif [x, y, ra, dec, radius].count(None) > 0:

            raise Exception('x, y, ra, dec and radius must all be specified')

        assert scale is None or 0 < scale[0] < scale[1]
        assert angle is None or -np.pi <= angle[0] < angle[1] <= np.pi

        self.bounds.scale = scale if scale is not None else self.bounds.scale
        self.bounds.angle = angle if angle is not None else self.bounds.angle

    def _sorted_triangles(self, pool):

        for i, c in enumerate(pool):
            for i, b in enumerate(pool[:i]):
                for a in pool[:i]:

                    yield a, b, c

    def _sorted_product_pairs(self, p, q):

        i_p = np.argsort(np.arange(len(p)))
        i_q = np.argsort(np.arange(len(q)))

        for _i_p, _i_q in sorted(product(i_p, i_q),
                                 key=lambda idxs: sum(idxs)):

            yield p[_i_p], q[_i_q]

    def _sorted_triangle_packages(self):

        i_xy_triangle_generator = self._sorted_triangles(self.i_xy)
        i_rd_triangle_generator = self._sorted_triangles(self.i_rd)

        i_xy_triangle_slice_generator = (tuple(
            islice(i_xy_triangle_generator, self.triangle_package_size)) for _ in count())
        i_rd_triangle_slice_generator = (list(
            islice(i_rd_triangle_generator, self.triangle_package_size)) for _ in count())

        for n in count(step=self.n_triangle_packages):

            i_xy_triangle_slice = tuple(
                filter(
                    None,
                    islice(i_xy_triangle_slice_generator,
                           self.n_triangle_packages)))
            i_rd_triangle_slice = tuple(
                filter(
                    None,
                    islice(i_rd_triangle_slice_generator,
                           self.n_triangle_packages)))

            if not len(i_xy_triangle_slice) and not len(i_rd_triangle_slice):
                return

            i_xy_triangle_generator2 = self._sorted_triangles(self.i_xy)
            i_rd_triangle_generator2 = self._sorted_triangles(self.i_rd)

            i_xy_triangle_cum = filter(None, (tuple(
                islice(i_xy_triangle_generator2, self.triangle_package_size)) for _ in range(n)))
            i_rd_triangle_cum = filter(None, (tuple(
                islice(i_rd_triangle_generator2, self.triangle_package_size)) for _ in range(n)))

            for i_xy_triangles, i_rd_triangles in chain(
                    filter(
                        None,
                        chain(*zip_longest(  # alternating chain
                            product(i_xy_triangle_slice, i_rd_triangle_cum),
                            product(i_xy_triangle_cum, i_rd_triangle_slice)))),
                    self._sorted_product_pairs(i_xy_triangle_slice,
                                               i_rd_triangle_slice)):
                yield np.array(i_xy_triangles), np.array(i_rd_triangles)

    def _get_triangle_angles(self, triangles):

        sidelengths = np.sqrt(
            np.power(triangles[:, (1, 0, 0)] - triangles[:, (2, 2, 1)],
                     2).sum(axis=2))

        # law of cosines
        angles = np.power(sidelengths[:, ((1, 2), (0, 2), (0, 1))],
                          2).sum(axis=2)
        angles -= np.power(sidelengths[:, (0, 1, 2)], 2)
        angles /= 2 * sidelengths[:, ((1, 2), (0, 2), (0, 1))].prod(axis=2)

        return np.arccos(angles)

    def _solve_for_matrices(self, xy_triangles, rd_triangles):

        n = len(xy_triangles)

        A = xy_triangles - np.mean(xy_triangles, axis=1).reshape(n, 1, 2)
        b = rd_triangles - np.mean(rd_triangles, axis=1).reshape(n, 1, 2)

        matrices = [
            np.linalg.lstsq(Ai, bi, rcond=None)[0].T for Ai, bi in zip(A, b)
        ]

        return np.array(matrices)

    def _extract_parameters(self, xy_triangles, rd_triangles, matrices):

        parameters = []

        for xy_com, rd_com, matrix in zip(  # com -> center-of-mass
                xy_triangles.mean(axis=1), rd_triangles.mean(axis=1),
                matrices):

            cos_dec = np.cos(np.deg2rad(rd_com[1]))
            coordinates = (self.bounds.xy - xy_com).dot(matrix)
            coordinates = coordinates / np.array([cos_dec, 1]) + rd_com

            wcs = WCS2.from_matrix(*xy_com, *rd_com, matrix)

            parameters.append(
                (*coordinates, np.log(wcs.scale), np.deg2rad(wcs.angle)))

        return parameters

    def _get_bounds_mask(self, parameters):

        i = np.ones(len(parameters), dtype=bool)
        parameters = np.array(parameters)

        if self.bounds.radius is not None:

            i *= angular_separation(
                *np.deg2rad(self.bounds.rd),
                *zip(*np.deg2rad(parameters[:, (0, 1)]))) <= np.deg2rad(
                    self.bounds.radius)

        if self.bounds.scale is not None:

            i *= self.bounds.scale[0] <= parameters[:, 2]
            i *= parameters[:, 2] <= self.bounds.scale[1]

        if self.bounds.angle is not None:

            i *= self.bounds.angle[0] <= parameters[:, 3]
            i *= parameters[:, 3] <= self.bounds.angle[1]

        return i

    def __call__(self, minimum_matches=4, ratio_superiority=1, timeout=60):
        '''Start the alogrithm.

        Can be run multiple times with different arguments to relax the
        restrictions.

        Example
        --------
        cm = CoordinateMatch(xy, rd)

        lkwargs = [{
            minimum_matches = 20,
            ratio_superiority = 5,
            timeout = 10
        },{
            timeout = 60
        }

        for i, kwargs in enumerate(lkwargs):
            try:
                i_xy, i_rd = cm(**kwargs)
            except TimeoutError:
                continue
            except StopIteration:
                print('Failed, no more stars.')
            else:
                print('Success with kwargs[%d].' % i)
        else:
            print('Failed, timeout.')
        '''

        self.parameters = list(
        ) if self.parameters is None else self.parameters

        t0 = time.time()

        while time.time() - t0 < timeout:

            # get triangles and derive angles

            i_xy_triangles, i_rd_triangles = next(
                self.triangle_package_generator)

            xy_angles = self._get_triangle_angles(self._xy[i_xy_triangles])
            rd_angles = self._get_triangle_angles(self._rd[i_rd_triangles])

            # sort triangle vertices based on angles

            i = np.argsort(xy_angles, axis=1)
            i_xy_triangles = np.take_along_axis(i_xy_triangles, i, axis=1)
            xy_angles = np.take_along_axis(xy_angles, i, axis=1)

            i = np.argsort(rd_angles, axis=1)
            i_rd_triangles = np.take_along_axis(i_rd_triangles, i, axis=1)
            rd_angles = np.take_along_axis(rd_angles, i, axis=1)

            # match triangles

            matches = KDTree(xy_angles).query_ball_tree(
                KDTree(rd_angles), r=self.maximum_angle_distance)
            matches = np.array([(_i_xy, _i_rd)
                                for _i_xy, _li_rd in enumerate(matches)
                                for _i_rd in _li_rd])

            if not len(matches):
                continue

            i_xy_triangles = list(i_xy_triangles[matches[:, 0]])
            i_rd_triangles = list(i_rd_triangles[matches[:, 1]])

            # get parameters of wcs solutions

            matrices = self._solve_for_matrices(
                self._xy[np.array(i_xy_triangles)],
                self._rd[np.array(i_rd_triangles)])

            parameters = self._extract_parameters(
                self.xy[np.array(i_xy_triangles)],
                self.rd[np.array(i_rd_triangles)], matrices)

            # apply bounds if any

            if any([self.bounds.radius, self.bounds.scale, self.bounds.angle]):

                mask = self._get_bounds_mask(parameters)

                i_xy_triangles = np.array(i_xy_triangles)[mask].tolist()
                i_rd_triangles = np.array(i_rd_triangles)[mask].tolist()
                parameters = np.array(parameters)[mask].tolist()

            # normalize parameters

            normalization = [
                getattr(self.normalizations, v)
                for v in ('ra', 'dec', 'scale', 'angle')
            ]
            normalization[0] *= np.cos(np.deg2rad(self.rd[:, 1].mean(axis=0)))
            parameters = list(parameters / np.array(normalization))

            # match parameters

            neighbours = KDTree(parameters).query_ball_tree(
                KDTree(self.parameters + parameters), r=self.distance_factor)
            neighbours = np.array([
                (i, j) for i, lj in enumerate(neighbours, len(self.parameters))
                for j in lj
            ])
            neighbours = list(
                neighbours[(np.diff(neighbours, axis=1) < 0).flatten()])

            if not len(neighbours):
                continue

            self.i_xy_triangles += i_xy_triangles
            self.i_rd_triangles += i_rd_triangles
            self.parameters += parameters
            self.neighbours.add_edges_from(neighbours)

            # get largest neighborhood

            communities = list(connected_components(self.neighbours))
            c1 = np.array(list(max(communities, key=len)))
            i = np.unique(np.array(self.i_xy_triangles)[c1].flatten(),
                          return_index=True)[1]

            if ratio_superiority > 1 and len(communities) > 1:

                communities.remove(set(c1))
                c2 = np.array(list(max(communities, key=len)))
                _i = np.unique(np.array(self.i_xy_triangles)[c2].flatten())

                if len(i) / len(_i) < ratio_superiority:
                    continue

            if len(i) >= minimum_matches:
                break

        else:

            raise TimeoutError

        i_xy = np.array(self.i_xy_triangles)[c1].flatten()[i]
        i_rd = np.array(self.i_rd_triangles)[c1].flatten()[i]

        return list(zip(i_xy, i_rd))
