"""
rotation_generator.py: A library to generate small 3D rotation matrices

The class generates a 3x3 matrix describing a general rotation in 3D
space, where each element relevant to the rotation is defined relative
to angles (in radians) of rotation for roll, pitch, and yaw, all of
which have 0 (inclusive) minima and user-specified maxima (inclusive).

This is intended to create "small" rotations, so we recommend
limiting the input angle maximum to π/4 (~0.785398) radians.
"""

import numpy as np
from scipy.spatial.transform import Rotation
import random
from typing import Tuple

class RotationGenerator:

    def __init__(self, angle_max: float):
        """
        init function for the RotationGenerator class.

        Args:
            angle_max: The maximum radian value for all randomly
	    	       generated angles of rotation.
        """
        self.angle_max = angle_max

    def generate_simple_rotation(self) -> Tuple[np.array, np.array]:
        """
        A method to generate a random simple 3D rotation matrix.

        A simple 3D rotation is a rotation by one angle theta about one
        axis, which here is also randomly chosen.
        
        Args:
            as_quat: Determines whether or not to return the rotation matrix
                     as a quaternion
        Returns:
            rot_mat: A 3x3 matrix describing a simple rotation transformation.
            rot_quat: A 4-vector quaternion representation of the rotation.
        """
        theta = random.uniform(0, self.angle_max)
        cos_val, sin_val = np.cos(theta), np.sin(theta)

        # Rotation about x
        r_x = np.array([[1,       0,        0],
                        [0, cos_val, -sin_val],
                        [0, sin_val, cos_val]])

        # Rotation about y
        r_y = np.array([[cos_val,  0, sin_val],
                        [0      ,  1,       0],
                        [-sin_val, 0, cos_val]])

        # Rotation about z
        r_z = np.array([[cos_val, -sin_val, 0],
                        [sin_val,  cos_val, 0],
                        [0      ,  0      , 1]])
        
        rots = [r_x, r_y, r_z]
        
        #rand_idx = random.choice([0, 1, 2])
        rot_mat = r_z #rots[rand_idx]
        rot_quat = Rotation.from_matrix(rot_mat).as_quat()

        return rot_mat, rot_quat


    def generate_rotation(self) -> Tuple[np.array, np.array]:
        """
        A method to generate a random general 3D rotation matrix.

        A general 3D rotation matrix is a rotation consisting of roll,
        pitch, and yaw angles about all three axes. It can be broken
        down into a matrix product of rotations about x by roll angle
        gamma, about y by pitch angle beta, and about z by yaw angle
        alpha.

        Returns:
            rot_mat: A 3x3 matrix describing a general rotation transformation.
            rot_quat: A 4-vector quaternion representation of the rotation.
        """
        rot_xyz = np.random.uniform(low=0.0, high=self.angle_max, size=(3,))
        r = Rotation.from_euler("xyz", rot_xyz)
        rot_quat = r.as_quat()
        rot_mat = r.as_matrix()
        return rot_mat, rot_quat
