import sys

import pybullet as p
import pybullet_data

from . import image
from . import objects
from .camera import Camera, make_obs

sys.path.insert(1, '../')
from .rotation_generator import RotationGenerator
import os
import numpy as np
from scipy.spatial.transform import Rotation as R


class RotationComparator(object):

    def __init__(self, save_dir: str, axis_urdf_path: str):
        self.this_camera = Camera(
            image_size=(480, 480),
            near=0.01,
            far=10.0,
            fov_w=69.40
        )

        self.save_dir = save_dir
        self.axis_path = axis_urdf_path
        self.max_rot_per_axis = 30
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir + "/rgb/", exist_ok=True)

    def compare_rotations(self, gt_rot, pred_rot, idx):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)  # dont need gravity, objects stay where they are
        rot_gen = RotationGenerator(self.max_rot_per_axis * np.pi / 180)

        rand_orientation = objects.gen_obj_orientation(num_scene=1, num_obj=1)[0]

        # Load floor
        p.loadURDF("plane.urdf")
        current_file_num = 0
        print("Start comparing the applied rotation.")

        # Load current object
        cur_id = p.loadURDF(
            fileName=self.axis_path,
            basePosition=[0, 0, 0.1],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            globalScaling=1,
        )
        # Apply a random rotation to the object
        p.resetBasePositionAndOrientation(
            cur_id,
            posObj=[0, 0, 0.1],
            ornObj=p.getQuaternionFromEuler(
                [rand_orientation[0],
                 rand_orientation[1],
                 rand_orientation[2]]
            )
        )

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.0, 0.0, 0.1],
            distance=0.3,
            yaw=0,
            pitch=-45,
            roll=0,
            upAxisIndex=2
        )
        rbg_obs, depth_obs, mask_obs = make_obs(self.this_camera, view_matrix)
        rgb_name = self.save_dir + "/rgb/" + str(idx) + "_before_rgb.png"
        image.write_rgb(rbg_obs.astype(np.uint8), rgb_name)

        # apply rotation matrix to object's position and orientation
        curRotMat = R.from_euler('zyx', rand_orientation, degrees=False).as_matrix()
        newRotMat_gt = curRotMat @ gt_rot
        newQuat_gt = R.from_matrix(newRotMat_gt).as_quat()

        newRotMat_pred = curRotMat @ pred_rot
        newQuat_pred = R.from_matrix(newRotMat_pred).as_quat()

        # Reset the object's position with respect to new values
        p.resetBasePositionAndOrientation(
            cur_id,
            posObj=[0,0,0.1],
            ornObj=newQuat_gt
        )
        # save an observation post-transformation matrix
        rbg_obs, depth_obs, mask_obs = make_obs(self.this_camera, view_matrix)
        rgb_name = self.save_dir + "/rgb/" + str(idx) + "_after_gt_rgb.png"
        image.write_rgb(rbg_obs.astype(np.uint8), rgb_name)


        p.resetBasePositionAndOrientation(
            cur_id,
            posObj=[0, 0, 0.1],
            ornObj=newQuat_pred
        )

        rbg_obs, depth_obs, mask_obs = make_obs(self.this_camera, view_matrix)
        rgb_name = self.save_dir + "/rgb/" + str(idx) + "_after_pred_rgb.png"
        image.write_rgb(rbg_obs.astype(np.uint8), rgb_name)

        p.removeBody(cur_id)
        p.disconnect()


def main():
    comp_rot = RotationComparator(
        save_dir="./",
        axis_urdf_path="./coordinate.urdf",
    )

    gt_rot = R.from_euler('xyz',[0,0,20],degrees=True).as_matrix()
    pred_rot = R.from_euler('xyz',[0,0,37],degrees=True).as_matrix()

    idx = 23

    comp_rot.compare_rotations(gt_rot, pred_rot, idx)

if __name__ == '__main__':
    main()
