import numpy as np
import pybullet as p
from typing import List


def gen_obj_orientation(num_scene: int, num_obj: int) -> List[List[float]]:
    """
    Generates a list of random unique orientations for each object in a scene.

    Args:
        num_scene: The number of scenes in the given sim.
        num_obj: The number of objects in the given scene.
    Returns:
        obj_orientations: A list containing each object's orientation defined
                          as a 3-vector of X, Y, and Z Euler angles in radians.
                          X describes the roll angle, Y the pitch angle, and Z
                          the yaw angle.
    """
    obj_orientations = []
    num_ori = num_scene * num_obj
    list_roll = np.random.choice(360, num_ori, replace=False)
    list_pitch = np.random.choice(360, num_ori, replace=False)
    list_yaw = np.random.choice(360, num_ori, replace=False)

    # Degrees to radians
    list_roll = [x/180*np.pi for x in list_roll]
    list_pitch = [x / 180 * np.pi for x in list_pitch]
    list_yaw = [x / 180 * np.pi for x in list_yaw]

    for i in range(num_ori):
        obj_orientations.append(
            [list_roll[i],
             list_pitch[i],
             list_yaw[i]]
        )
    return obj_orientations


def load_obj(
    obj_foldernames: List[str],
    positions: List[List[float]]) -> List[int]:
    """
    Load objects into the scene.

    Args:
        obj_foldernames: Folder names for the objects to be loaded into the
                         scene.
        positions: The initial positions of each of the objects to be loaded
                   into the scene.

    Returns:
        obj_ids: The IDs of objects loaded. If all objects are loaded
                 successfully, return is positive definite.
    """
    obj_ids = []
    num_obj = len(obj_foldernames)
    for i in range(num_obj):
        cur_id = p.loadURDF(
            fileName="./YCB_subsubset/" + obj_foldernames[i] + "/obj.urdf",
            basePosition=positions[i],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            globalScaling=1,
        )
        obj_ids.append(cur_id)
    return obj_ids


def reset_obj(
    obj_ids: List[int],
    positions: List[List[float]],
    orientations: List[List[float]],
    scene_id: int) -> None:
    """
    Resets objects in the simulation.

    Args:
        obj_ids: A list of the IDs of objects present in the scene.
        positions: A list of the existing objects' positions, each as 3-element
                   vectors of Euclidean x, y, z coordinates.
        orientations: A list of the existing objects' orientations.
        scene_id: The ID for this scene.
    """
    num_obj = len(obj_ids)
    np.random.seed(scene_id)
    # Currently only one object (to fix, choice(5,5,replace=False))
    position_index = np.random.choice(1, 5, replace=True)
    for i in range(num_obj):
        cur_orientation = orientations[scene_id * num_obj + i]
        p.resetBasePositionAndOrientation(
            obj_ids[i],
            posObj=positions[position_index[i]],
            ornObj=p.getQuaternionFromEuler(
                [cur_orientation[0],
                 cur_orientation[1],
                 cur_orientation[2]]
            )
        )
    # Drop objects on the floor
    # for tick in range(1):
    #     p.stepSimulation()
    return
