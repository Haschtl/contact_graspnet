# -*- coding: utf-8 -*-
"""Helper classes and functions to sample grasps for a given object mesh."""

from __future__ import print_function

import os
import numpy as np
from tqdm import tqdm
import trimesh
from PandaGripper import PandaGripper
from DavidSimpleGripper import DavidSimpleGripper

class Object(object):
    """Represents a graspable object."""

    def __init__(self, filename):
        """Constructor.

        :param filename: Mesh to load
        :param scale: Scaling factor
        """
        self.mesh = trimesh.load(filename)
        self.scale = 1.0

        # print(filename)
        self.filename = filename
        if isinstance(self.mesh, list):
            # this is fixed in a newer trimesh version:
            # https://github.com/mikedh/trimesh/issues/69
            print("Warning: Will do a concatenation")
            self.mesh = trimesh.util.concatenate(self.mesh)

        self.collision_manager = trimesh.collision.CollisionManager()
        self.collision_manager.add_object('object', self.mesh)

    def rescale(self, scale=1.0):
        """Set scale of object mesh.

        :param scale
        """
        self.scale = scale
        self.mesh.apply_scale(self.scale)

    def resize(self, size=1.0):
        """Set longest of all three lengths in Cartesian space.

        :param size
        """
        self.scale = size / np.max(self.mesh.extents)
        self.mesh.apply_scale(self.scale)

    def in_collision_with(self, mesh, transform):
        """Check whether the object is in collision with the provided mesh.

        :param mesh:
        :param transform:
        :return: boolean value
        """
        return self.collision_manager.in_collision_single(mesh, transform=transform)


def create_gripper(name, configuration=None, root_folder=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
    """Create a gripper object.

    Arguments:
        name {str} -- name of the gripper

    Keyword Arguments:
        configuration {list of float} -- configuration (default: {None})
        root_folder {str} -- base folder for model files (default: {''})

    Raises:
        Exception: If the gripper name is unknown.

    Returns:
        [type] -- gripper object
    """
    if name.lower() == 'panda':
        return PandaGripper(q=configuration, root_folder=root_folder)
    if name.lower() == 'david-simple':
        return DavidSimpleGripper(q=configuration, root_folder=root_folder)
    else:
        raise Exception("Unknown gripper: {}".format(name))


def in_collision_with_gripper(object_mesh, gripper_transforms, gripper_name, silent=False):
    """Check collision of object with gripper.

    Arguments:
        object_mesh {trimesh} -- mesh of object
        gripper_transforms {list of numpy.array} -- homogeneous matrices of gripper
        gripper_name {str} -- name of gripper

    Keyword Arguments:
        silent {bool} -- verbosity (default: {False})

    Returns:
        [list of bool] -- Which gripper poses are in collision with object mesh
    """
    manager = trimesh.collision.CollisionManager()
    manager.add_object('object', object_mesh)
    gripper_meshes = [create_gripper(gripper_name).hand]
    min_distance = []
    for tf in tqdm(gripper_transforms, disable=silent):
        min_distance.append(np.min([manager.min_distance_single(
            gripper_mesh, transform=tf) for gripper_mesh in gripper_meshes]))

    return [d == 0 for d in min_distance], min_distance

def grasp_contact_location(transforms, successfuls, collisions, object_mesh, gripper_name='panda', silent=False):
    """Computes grasp contacts on objects and normals, offsets, directions

    Arguments:
        transforms {[type]} -- grasp poses
        collisions {[type]} -- collision information
        object_mesh {trimesh} -- object mesh

    Keyword Arguments:
        gripper_name {str} -- name of gripper (default: {'panda'})
        silent {bool} -- verbosity (default: {False})

    Returns:
        list of dicts of contact information per grasp ray
    """
    res = []
    gripper = create_gripper(gripper_name)
    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
            object_mesh, scale_to_box=True)
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)
    for p, colliding, outcome in tqdm(zip(transforms, collisions, successfuls), total=len(transforms), disable=silent):
        contact_dict = {}
        contact_dict['collisions'] = 0
        contact_dict['valid_locations'] = 0
        contact_dict['successful'] = outcome
        contact_dict['grasp_transform'] = p
        contact_dict['contact_points'] = []
        contact_dict['contact_directions'] = []
        contact_dict['contact_face_normals'] = []
        contact_dict['contact_offsets'] = []

        if colliding:
            contact_dict['collisions'] = 1
        else:
            ray_origins, ray_directions = gripper.get_closing_rays_contact(p)

            locations, index_ray, index_tri = intersector.intersects_location(
                ray_origins, ray_directions, multiple_hits=False)

            if len(locations) > 0:
                # this depends on the width of the gripper
                valid_locations = np.linalg.norm(ray_origins[index_ray]-locations, axis=1) <= 2.0*gripper.q

                if sum(valid_locations) > 1:
                    contact_dict['valid_locations'] = 1
                    contact_dict['contact_points'] = locations[valid_locations]
                    contact_dict['contact_face_normals'] = object_mesh.face_normals[index_tri[valid_locations]]
                    contact_dict['contact_directions'] = ray_directions[index_ray[valid_locations]]
                    contact_dict['contact_offsets'] = np.linalg.norm(ray_origins[index_ray[valid_locations]] - locations[valid_locations], axis=1)
                    # dot_prods = (contact_dict['contact_face_normals'] * contact_dict['contact_directions']).sum(axis=1)
                    # contact_dict['contact_cosine_angles'] = np.cos(dot_prods)
                    res.append(contact_dict)
                
    return res
