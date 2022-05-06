
import os
import numpy as np
import pickle
import trimesh
import yaml
import trimesh.transformations as tra
from .Gripper import Gripper


class DavidSimpleGripper(Gripper):
    """An object representing a David parallel-yaw gripper."""

    def __init__(self, q=None, root_folder=os.path.join(os.path.dirname(os.path.realpath(__file__)), "david")):
        """Create a David parallel-yaw gripper object.

        Keyword Arguments:
            q {list of int} -- opening configuration (default: {None})
            root_folder {str} -- base folder for model files (default: {''})
        """
        Gripper.__init__(self, q)
        self.joint_limits = [0.0, 0.04]
        self.__initQ(q)
        self.__initMeshes(root_folder)
        self.__initContactRays(root_folder)
        self.__initControlPoints(root_folder)

    def __initQ(self, q):
        self.default_pregrasp_configuration = 0.04
        if q is None:
            q = self.default_pregrasp_configuration

        self.q = q

    def __initMeshes(self, root_folder):
        fn_base = os.path.join(
            root_folder, 'hand.stl')
        fn_finger_l = os.path.join(
            root_folder, 'finger_l.stl')
        fn_finger_r = os.path.join(
            root_folder, 'finger_r.stl')

        self.base = trimesh.load(fn_base)
        finger_l = trimesh.load(fn_finger_l)
        finger_r = trimesh.load(fn_finger_r)

        # transform fingers relative to the base
        finger_l.apply_transform(tra.euler_matrix(0, 0, np.pi))
        finger_l.apply_translation([+self.q, 0, 0.0584])
        finger_r.apply_translation([-self.q, 0, 0.0584])
        self.fingers = [finger_l, finger_r]

    def __initContactRays(self, root_folder):
        with open(os.path.join(root_folder, 'gripper_coords.yml'), 'rb') as f:
            # self.finger_coords = pickle.load(f, encoding='latin1')
            data_loaded = yaml.safe_load(f)
        for key in data_loaded:
            data_loaded[key] = np.array(data_loaded[key])
        finger_coords = data_loaded

        finger_direction = finger_coords['gripper_right_center_flat'] - \
            finger_coords['gripper_left_center_flat']
        self.contact_ray_origins.append(
            np.r_[finger_coords['gripper_left_center_flat'], 1])
        self.contact_ray_origins.append(
            np.r_[finger_coords['gripper_right_center_flat'], 1])
        self.contact_ray_directions.append(
            finger_direction / np.linalg.norm(finger_direction))
        self.contact_ray_directions.append(-finger_direction /
                                           np.linalg.norm(finger_direction))

        self.contact_ray_origins = np.array(self.contact_ray_origins)
        self.contact_ray_directions = np.array(self.contact_ray_directions)

    def __initControlPoints(self, root_folder):
        self.control_points = np.load(os.path.join(
            root_folder, 'control_points.npy'))[:, :3]
