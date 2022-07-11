from camera import Camera, save_obs
import objects
import pybullet_data
import pybullet as p
import sys

sys.path.insert(1, './')
from rotation_generator import RotationGenerator
import os
from typing import List, Dict, Tuple
import numpy as np
import re
from scipy.spatial.transform import Rotation as R


class DatasetGenerator(object):
    """
    A class which generates the data set for our project. The data set consists
    of scenes (pairs of images) each labeled by a rotation matrix. In a given
    scene, an object is dropped from an arbitrary height onto to the ground in
    a pybullet physics simulation. At that point, an observation is made (a
    picture is taken). Then, a randomly generated rotation matrix is applied to
    the object and another observation is made. Our model will be attempting to
    regress the rotation matrix that caused the transformation of the object
    from the first observation to the second observations.
    """

    def __init__(self,
                 training_scenes: int,
                 obj_foldernames: List[str],
                 obj_positions: List[List[float]],
                 dataset_dir: str,
                 max_rot_per_axis: int):
        """
        Initializes the DatasetGenerator class.

        Args:
            training_scenes: The number of scenes we'd like to give our model
                             to train on.
            obj_foldernames: The names of each object folder (located in
                             the YCB_subsubset directory of dataset_generator
                             folder).
            obj_positions: A list of the initial positions for each object,
                           each given as 3-vectors of Euclidean x, y, z
                           coordinates.
            dataset_dir: The directory we'd like to save our training examples
                         to.
            max_rot_per_axis: When rotating the objects, the max angle to rotate along each axis, in degrees.
        """
        self.this_camera = Camera(
            image_size=(64, 64),
            near=0.01,
            far=10.0,
            fov_w=69.40
        )
        """ Training scenes from init, how many pictures to take"""
        self.training_scenes = training_scenes
        self.obj_foldernames = [fn for fn in obj_foldernames]
        self.obj_positions = obj_positions
        self.obj_orientations = objects.gen_obj_orientation(
            num_scene=self.training_scenes,
            num_obj=1
        )
        self.max_rot_per_axis = max_rot_per_axis

        self.dataset_dir = dataset_dir
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(dataset_dir + "/rgb/", exist_ok=True)
        os.makedirs(dataset_dir + "/gt/", exist_ok=True)
        os.makedirs(dataset_dir + "/depth/", exist_ok=True)
        os.makedirs(dataset_dir + "/rotations/", exist_ok=True)

        self.rot_file = dataset_dir + "/rotations/rotations.csv"

    def generate_dataset(self) -> Dict[int, Tuple[np.array, Dict[str, List[str]]]]:
        """
        Generates the dataset of our project. It is saved to self.dataset_dir
        """
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)  # dont need gravity, objects stay where they are
        rot_gen = RotationGenerator(self.max_rot_per_axis * np.pi / 180)

        # Load floor
        p.loadURDF("plane.urdf")
        current_file_num = 0
        print("Start generating the training set.")

        with open(self.rot_file, 'a') as fd:
            for j in range(0, len(self.obj_foldernames)):

                # for each object folder in the list, generate training scenes
                current_obj = [self.obj_foldernames[j]]

                # get new orientations
                self.obj_orientations = objects.gen_obj_orientation(
                    num_scene=self.training_scenes,
                    num_obj=1
                )

                # Load current object
                obj_ids = objects.load_obj(current_obj, self.obj_positions)

                for i in range(0, self.training_scenes):
                    print(f'==> {i + 1} / {self.training_scenes}')
                    # Apply a random rotation to the object
                    objects.reset_obj(
                        obj_ids,
                        self.obj_positions,
                        self.obj_orientations,
                        scene_id=i
                    )

                    # save an observation pre-transformation matrix
                    rgb1, mask1, depth1 = save_obs(self.dataset_dir, self.this_camera, current_file_num, "before")
                    # collect current position and orientation info
                    objPos, objOrn = p.getBasePositionAndOrientation(obj_ids[0])

                    # generate a random 3D rotation quaternion
                    # rot_mat, rot_quat = rot_gen.generate_simple_rotation()  # Rotate only about X-axis
                    rot_mat, rot_quat = rot_gen.generate_rotation()  # Rotate about all 3 axes

                    # apply rotation matrix to object's position and orientation
                    curRotMat = np.array(p.getMatrixFromQuaternion(objOrn)).reshape(3, 3)
                    newRotMat = curRotMat @ rot_mat
                    newQuat = R.from_matrix(newRotMat).as_quat()

                    # Reset the object's position with respect to new values
                    p.resetBasePositionAndOrientation(
                        obj_ids[0],  # currently only works for the banana
                        posObj=objPos,
                        ornObj=newQuat
                    )

                    # Save rot_matrix after making it flat
                    rot_flat = str(rot_mat.flatten())
                    rot_clean_str = " "
                    rot_clean_str = rot_clean_str.join(rot_flat.split())
                    rot_clean_str = rot_clean_str[2:-1]
                    fd.write(rot_clean_str + "\n")

                    # save an observation post-transformation matrix
                    rgb2, mask2, depth2 = save_obs(self.dataset_dir, self.this_camera, current_file_num, "after")
                    current_file_num += 1

                p.removeBody(obj_ids[0])
        p.disconnect()


def main():
    data_gen = DatasetGenerator(
        training_scenes=5,
        # obj_foldernames=["004_sugar_box", "005_tomato_soup_can", "007_tuna_fish_can", "011_banana", "024_bowl"],
        obj_foldernames=["006_mustard_bottle", "008_pudding_box", "009_gelatin_box", "010_potted_meat_can", "019_pitcher_base"],
        obj_positions=[[0.0, 0.0, 0.1]],
        dataset_dir="../dataset/test",
        max_rot_per_axis=30
    )
    data_gen.generate_dataset()


if __name__ == '__main__':
    main()
