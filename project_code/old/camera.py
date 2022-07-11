import math

import numpy as np
import pybullet as p
from typing import List, Tuple
import image


class Camera(object):
    """
    Class to define a camera. Modified from Zhenjia Xu's camera setting.
    """
    def __init__(self,
        image_size: Tuple[int],
        near: float,
        far: float,
        fov_w: float):
        """
        Creates a virtual camera from the given parameters.

        Args:
            image_size: (height, width) of the images our camera will take.
            near: Value of the near plane.
            far: Value of the far plane.
            fov_w: Field of view in width direction in degrees.
        """
        super().__init__()

        # Set from input args
        self.image_size = image_size
        self.near = near
        self.far = far
        self.fov_width = fov_w
        self.focal_length = (float(self.image_size[1]) / 2) / np.tan((np.pi * self.fov_width / 180) / 2)
        self.fov_height = (math.atan((float(self.image_size[0]) / 2) / self.focal_length) * 2 / np.pi) * 180
        self.intrinsic_matrix, self.projection_matrix = self.compute_camera_matrix()


    def compute_camera_matrix(self) -> Tuple[np.array, List[float]]:
        """
        Computes intrinsic and projection matrices from instance variables in
        Camera class.

        Returns:
            intrinsic_matrix: 3x3 numpy array corresponding to the intrinsic
                              camera matrix.
            projection_matrix: A list of 16 floats, representing a 4x4
                               projection matrix for the camera view.
        """
        intrinsic_matrix = np.eye(3)
        intrinsic_matrix = intrinsic_matrix.astype('float64')
        radF_y = ((np.pi * self.fov_height) / 180)
        radF_x = (np.pi * self.fov_width / 180)
        f_y = float(self.image_size[0])/(2.0*np.tan(radF_y/2.0))
        f_x = float(self.image_size[1])/(2.0*np.tan(radF_x/2.0))
        intrinsic_matrix[0][0] = self.focal_length
        intrinsic_matrix[1][1] = self.focal_length
        intrinsic_matrix[0][2] = float(self.image_size[1])/2.0
        intrinsic_matrix[1][2] = float(self.image_size[0])/2.0
        ar = float(self.image_size[1]/self.image_size[0])
        projection_matrix = p.computeProjectionMatrixFOV(self.fov_width, ar, self.near, self.far)
        return intrinsic_matrix, projection_matrix


def cam_view2pose(cam_view_matrix: List[float]) -> np.array:
    """
    Converts a camera view matrix to pose matrix.

    Args:
        cam_view_matrix: A list of 16 floats representing a 4x4 camera
                         projection matrix.
    Returns:
        cam_pose_matrix: 4x4 numpy array corresponding to the 16 float list.
    """
    cam_pose_matrix = np.linalg.inv(np.array(cam_view_matrix).reshape(4, 4).T)
    cam_pose_matrix[:, 1:3] = -cam_pose_matrix[:, 1:3]
    return cam_pose_matrix


def make_obs(
    camera: Camera,
    view_matrix: List[float]) -> (np.array, np.array, np.array):
    """
    Uses a camera to make an observation and return RGB, depth, and instance
    level segmentation mask observations.

    Args:
        camera: The Camera instance making the observations.
        view_matrix: The camera's 4x4 view matrix (represented as a list of 16
                     floats).
    Returns:
        rgb_obs: The RGB observation corresponding to the current scene.
        depth_obs: Depth information corresponding to the current scene.
        mask_obs: The current scene's mask.
    """
    obs = p.getCameraImage(
        width=camera.image_size[1],
        height=camera.image_size[0],
        viewMatrix=view_matrix,
        projectionMatrix=camera.projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        shadow=1,
    )

    need_convert = False
    if type(obs[2]) is tuple:
        need_convert = True

    if need_convert:
        rgb_pixels = np.asarray(obs[2]).reshape(camera.image_size[0], camera.image_size[1], 4)
        rgb_obs = rgb_pixels[:, :, :3]
        z_buffer = np.asarray(obs[3]).reshape(camera.image_size[0], camera.image_size[1])
        depth_obs = camera.far * camera.near / (camera.far - (camera.far - camera.near) * z_buffer)
        mask_obs = np.asarray(obs[4]).reshape(camera.image_size[0], camera.image_size[1])
    else:
        rgb_obs = obs[2][:, :, :3]
        depth_obs = camera.far * camera.near / (camera.far - (camera.far - camera.near) * obs[3])
        mask_obs = obs[4]

    mask_obs[mask_obs == -1] = 0  # label empty space as 0 (plane)
    return rgb_obs, depth_obs, mask_obs


def save_obs(
    dataset_dir: str,
    camera: Camera,
    scene_id: int,
    trans_applied: str,
    is_valset=False) -> Tuple[str, str]:
    """
    Saves RGB, depth, and instance level segmentation mask as files.

    Args:
        dataset_dir: The directory to which we save observations.
        camera: The camera instance we're using.
        scene_id: The scene we're observing, used to index files to be saved.
        trans_applied: Whether or not the transformation matrix has already
                       been applied. Should be "before" (before transformation)
                       or "after" (after transformation).
    Returns:
        rgb_name: The name of the rgb file of the observation.
        mask_name: The name of the object mask corresponding to the observation.
    """
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=(0.0, 0.0, 0.0),
        distance = 0.7,
        yaw = 0,
        pitch = -25,
        roll = 0,
        upAxisIndex = 2
    )
    rbg_obs, depth_obs, mask_obs = make_obs(camera, view_matrix)
    rgb_name = dataset_dir+"/rgb/"+str(scene_id)+"_"+trans_applied+"_rgb.png"
    image.write_rgb(rbg_obs.astype(np.uint8), rgb_name)
    print(rgb_name) # TESTING
    mask_name = dataset_dir+"/gt/"+str(scene_id)+"_"+trans_applied+"_gt.png"
    image.write_mask(mask_obs, mask_name)
    print(mask_name) # TESTING
    return rgb_name, mask_name
