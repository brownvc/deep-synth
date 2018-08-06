import random
import numpy as np

from pyquaternion import Quaternion
from math_utils import Transform


class OBB(object):
    def __init__(self, center, half_widths, rotation_matrix):
        self._c = np.squeeze(center)
        self._h = np.squeeze(half_widths)
        self._R = rotation_matrix
        self._recompute_transforms()

    def _recompute_transforms(self):
        # local-to-world transform: takes [0,1]^3 points in OBB to world space points
        self._local_to_world = np.identity(4)
        for i in range(3):
            self._local_to_world[:3, i] = self._R[:, i] * self._h[i]
        self._local_to_world[:3, 3] = self._c

        # world-to-local transform: world space points within OBB to [0,1]^3
        self._world_to_local = np.identity(4)
        for i in range(3):
            self._world_to_local[i, :3] = self._R[:, i] * (1.0 / self._h[i])
        t_inv = - np.matmul(self._world_to_local[:3, :3], np.transpose(self._c))
        self._world_to_local[:3, 3] = np.squeeze(t_inv)

    @classmethod
    def from_local2world_transform(cls, transform):
        xform = Transform.from_mat4(transform)
        return cls(center=xform.translation, half_widths=xform.scale, rotation_matrix=xform.rotation.rotation_matrix)

    @classmethod
    def from_node(cls, node, aligned_dims):
        xform = Transform.from_mat4x4_flat_row_major(node.transform)
        return cls(center=xform.translation, half_widths=aligned_dims * xform.scale * 0.5,
                   rotation_matrix=xform.rotation.rotation_matrix)

    @property
    def half_extents(self):
        return self._h

    @property
    def rotation_matrix(self):
        return self._R

    @property
    def rotation_quat(self):
        return Quaternion(matrix=self._R)

    @property
    def dimensions(self):
        return 2.0 * self._h

    @property
    def half_dimensions(self):
        return self._h

    @property
    def centroid(self):
        return self._c

    @property
    def world2local(self):
        return self._world_to_local

    @property
    def local2world(self):
        return self._local_to_world

    def __repr__(self):
        return 'OBB: {c:' + str(self._c) + ',h:' + str(self._h) + ',R:' + str(self._R.tolist()) + '}'

    def transform_point(self, p):
        return np.matmul(self._world_to_local, np.append(p, [1], axis=0))[:3]

    def transform_direction(self, d):
        return np.matmul(np.transpose(self._R), d)

    def distance_to_point(self, p):
        if self.contains_point(p):
            return 0.0
        closest = self.closest_point(p)
        return np.linalg.norm(p - closest)

    def contains_point(self, p):
        p_local = np.matmul(self._world_to_local, np.append(p, [1], axis=0))[:3]
        bound = 1.0
        for i in range(3):
            if abs(p_local[i]) > bound:
                return False
        return True  # here only if all three coord within bounds

    def closest_point(self, p):
        d = p - self._c
        closest = np.copy(self._c)
        for i in range(3):
            closest += np.clip(self._R[:, i] * d, -self._h[i], self._h[i]) * self._R[:, i]
        return closest

    def sample(self):
        p = np.copy(self._c)
        for i in range(3):
            r = random.random() * 2.0 - 1.0
            p += r * self._h[i] * self._R[:, i]
        return p

    def project_to_axis(self, direction):
        """
        Projects this OBB onto the given 1D direction vector and returns projection interval [min, max].
        If vector is unnormalized, the output is scaled by the length of the vector
        :param direction: the axis on which to project this OBB
        :return: out_min - the minimum extent of this OBB along direction, out_max - the maximum extent
        """
        x = abs(np.dot(direction, self._R[0]) * self._h[0])
        y = abs(np.dot(direction, self._R[1]) * self._h[1])
        z = abs(np.dot(direction, self._R[2]) * self._h[2])
        p = np.dot(direction, self._c)
        out_min = p - x - y - z
        out_max = p + x + y + z
        return out_min, out_max

    def signed_distance_to_plane(self, plane):
        p_min, p_max = self.project_to_axis(plane.normal)
        p_min -= plane.d
        p_max -= plane.d
        if p_min * p_max <= 0.0:
            return 0.0
        return p_min if abs(p_min) < abs(p_max) else p_max

    def to_aabb(self):
        h_size = abs(self._R[0] * self._h[0]) + abs(self._R[1] * self._h[1]) + abs(self._R[2] * self._h[2])
        p_min = self._c - h_size
        p_max = self._c + h_size
        return p_min, p_max
