import provider
import trimesh.transformations as tra
import cv2
import copy
import numpy as np
import os

from ..pointcloud.tools import estimate_normals_cam_from_pc, regularize_pc_point_count
from ..util.geometry import distance_by_translation_point, farthest_points
from ..scene.scene_renderer import SceneRenderer


class PointCloudReader:
    """
    Class to load scenes, render point clouds and augment them during training

    Arguments:
        root_folder {str} -- acronym root folder
        batch_size {int} -- number of rendered point clouds per-batch

    Keyword Arguments:
        raw_num_points {int} -- Number of random/farthest point samples per scene (default: {20000})
        estimate_normals {bool} -- compute normals from rendered point cloud (default: {False})
        caching {bool} -- cache scenes in memory (default: {True})
        use_uniform_quaternions {bool} -- use uniform quaternions for camera sampling (default: {False})
        scene_obj_scales {list} -- object scales in scene (default: {None})
        scene_obj_paths {list} -- object paths in scene (default: {None})
        scene_obj_transforms {np.ndarray} -- object transforms in scene (default: {None})
        num_train_samples {int} -- training scenes (default: {None})
        num_test_samples {int} -- test scenes (default: {None})
        use_farthest_point {bool} -- use farthest point sampling to reduce point cloud dimension (default: {False})
        intrinsics {str} -- intrinsics to for rendering depth maps (default: {None})
        distance_range {tuple} -- distance range from camera to center of table (default: {(0.9,1.3)})
        elevation {tuple} -- elevation range (90 deg is top-down) (default: {(30,150)})
        pc_augm_config {dict} -- point cloud augmentation config (default: {None})
        depth_augm_config {dict} -- depth map augmentation config (default: {None})
    """

    def __init__(
        self,
        root_folder,
        batch_size=1,
        raw_num_points=20000,
        estimate_normals=False,
        caching=True,
        use_uniform_quaternions=False,
        scene_obj_scales=None,
        scene_obj_paths=None,
        scene_obj_transforms=None,
        num_train_samples=None,
        num_test_samples=None,
        use_farthest_point=False,
        intrinsics=None,
        distance_range=(0.9, 1.3),
        elevation=(30, 150),
        pc_augm_config=None,
        depth_augm_config=None
    ):
        self._root_folder = root_folder
        self._batch_size = batch_size
        self._raw_num_points = raw_num_points
        self._caching = caching
        self._num_train_samples = num_train_samples
        self._num_test_samples = num_test_samples
        self._estimate_normals = estimate_normals
        self._use_farthest_point = use_farthest_point
        self._scene_obj_scales = scene_obj_scales
        self._scene_obj_paths = scene_obj_paths
        self._scene_obj_transforms = scene_obj_transforms
        self._distance_range = distance_range
        self._pc_augm_config = pc_augm_config
        self._depth_augm_config = depth_augm_config

        self._current_pc = None
        self._cache = {}

        self._renderer = SceneRenderer(caching=True, intrinsics=intrinsics)

        if use_uniform_quaternions:
            quat_path = os.path.join(os.path.dirname(os.path.realpath(
                __file__)), 'uniform_quaternions/data2_4608.qua')
            quaternions = [l[:-1].split('\t')
                           for l in open(quat_path, 'r').readlines()]

            quaternions = [[float(t[0]),
                            float(t[1]),
                            float(t[2]),
                            float(t[3])] for t in quaternions]
            quaternions = np.asarray(quaternions)
            quaternions = np.roll(quaternions, 1, axis=1)
            self._all_poses = [tra.quaternion_matrix(q) for q in quaternions]
        else:
            self._cam_orientations = []
            self._elevation = np.array(elevation)/180.
            for az in np.linspace(0, np.pi * 2, 30):
                for el in np.linspace(self._elevation[0], self._elevation[1], 30):
                    self._cam_orientations.append(tra.euler_matrix(0, -el, az))
            self._coordinate_transform = tra.euler_matrix(
                np.pi/2, 0, 0).dot(tra.euler_matrix(0, np.pi/2, 0))

    def get_cam_pose(self, cam_orientation):
        """
        Samples camera pose on shell around table center 

        Arguments:
            cam_orientation {np.ndarray} -- 3x3 camera orientation matrix

        Returns:
            [np.ndarray] -- 4x4 homogeneous camera pose
        """

        distance = self._distance_range[0] + np.random.rand()*(
            self._distance_range[1]-self._distance_range[0])

        extrinsics = np.eye(4)
        extrinsics[0, 3] += distance
        extrinsics = cam_orientation.dot(extrinsics)

        cam_pose = extrinsics.dot(self._coordinate_transform)
        # table height
        cam_pose[2, 3] += self._renderer._table_dims[2]
        cam_pose[:3, :2] = -cam_pose[:3, :2]
        return cam_pose

    def _augment_pc(self, pc):
        """
        Augments point cloud with jitter and dropout according to config

        Arguments:
            pc {np.ndarray} -- Nx3 point cloud

        Returns:
            np.ndarray -- augmented point cloud
        """

        # not used because no artificial occlusion
        if 'occlusion_nclusters' in self._pc_augm_config and self._pc_augm_config['occlusion_nclusters'] > 0:
            pc = self.apply_dropout(pc,
                                    self._pc_augm_config['occlusion_nclusters'],
                                    self._pc_augm_config['occlusion_dropout_rate'])

        if 'sigma' in self._pc_augm_config and self._pc_augm_config['sigma'] > 0:
            pc = provider.jitter_point_cloud(pc[np.newaxis, :, :],
                                             sigma=self._pc_augm_config['sigma'],
                                             clip=self._pc_augm_config['clip'])[0]

        return pc[:, :3]

    def _augment_depth(self, depth):
        """
        Augments depth map with z-noise and smoothing according to config

        Arguments:
            depth {np.ndarray} -- depth map

        Returns:
            np.ndarray -- augmented depth map
        """

        if 'sigma' in self._depth_augm_config and self._depth_augm_config['sigma'] > 0:
            clip = self._depth_augm_config['clip']
            sigma = self._depth_augm_config['sigma']
            noise = np.clip(sigma*np.random.randn(*depth.shape), -clip, clip)
            depth += noise
        if 'gaussian_kernel' in self._depth_augm_config and self._depth_augm_config['gaussian_kernel'] > 0:
            kernel = self._depth_augm_config['gaussian_kernel']
            depth_copy = depth.copy()
            depth = cv2.GaussianBlur(depth, (kernel, kernel), 0)
            depth[depth_copy == 0] = depth_copy[depth_copy == 0]

        return depth

    def apply_dropout(self, pc, occlusion_nclusters, occlusion_dropout_rate):
        """
        Remove occlusion_nclusters farthest points from point cloud with occlusion_dropout_rate probability

        Arguments:
            pc {np.ndarray} -- Nx3 point cloud
            occlusion_nclusters {int} -- noof cluster to remove
            occlusion_dropout_rate {float} -- prob of removal

        Returns:
            [np.ndarray] -- N > Mx3 point cloud
        """
        if occlusion_nclusters == 0 or occlusion_dropout_rate == 0.:
            return pc

        labels = farthest_points(
            pc, occlusion_nclusters, distance_by_translation_point)

        removed_labels = np.unique(labels)
        removed_labels = removed_labels[np.random.rand(
            removed_labels.shape[0]) < occlusion_dropout_rate]
        if removed_labels.shape[0] == 0:
            return pc
        mask = np.ones(labels.shape, labels.dtype)
        for l in removed_labels:
            mask = np.logical_and(mask, labels != l)
        return pc[mask]

    def get_scene_batch(self, scene_idx=None, return_segmap=False, save=False):
        """
        Render a batch of scene point clouds

        Keyword Arguments:
            scene_idx {int} -- index of the scene (default: {None})
            return_segmap {bool} -- whether to render a segmap of objects (default: {False})
            save {bool} -- Save training/validation data to npz file for later inference (default: {False})

        Returns:
            [batch_data, cam_poses, scene_idx] -- batch of rendered point clouds, camera poses and the scene_idx
        """
        dims = 6 if self._estimate_normals else 3
        batch_data = np.empty(
            (self._batch_size, self._raw_num_points, dims), dtype=np.float32)
        cam_poses = np.empty((self._batch_size, 4, 4), dtype=np.float32)

        if scene_idx is None:
            scene_idx = np.random.randint(0, self._num_train_samples)

        obj_paths = [os.path.join(self._root_folder, p)
                     for p in self._scene_obj_paths[scene_idx]]
        mesh_scales = self._scene_obj_scales[scene_idx]
        obj_trafos = self._scene_obj_transforms[scene_idx]

        self.change_scene(obj_paths, mesh_scales, obj_trafos, visualize=False)

        batch_segmap, batch_obj_pcs = [], []
        for i in range(self._batch_size):
            # 0.005s
            pc_cam, pc_normals, camera_pose, depth = self.render_random_scene(
                estimate_normals=self._estimate_normals)

            if return_segmap:
                segmap, _, obj_pcs = self._renderer.render_labels(
                    depth, obj_paths, mesh_scales, render_pc=True)
                batch_obj_pcs.append(obj_pcs)
                batch_segmap.append(segmap)

            batch_data[i, :, 0:3] = pc_cam[:, :3]
            if self._estimate_normals:
                batch_data[i, :, 3:6] = pc_normals[:, :3]
            cam_poses[i, :, :] = camera_pose

        if save:
            K = np.array([[616.36529541, 0, 310.25881958], [
                         0, 616.20294189, 236.59980774], [0, 0, 1]])
            data = {'depth': depth, 'K': K,
                    'camera_pose': camera_pose, 'scene_idx': scene_idx}
            if return_segmap:
                data.update(segmap=segmap)
            np.savez('results/{}_acronym.npz'.format(scene_idx), data)

        if return_segmap:
            return batch_data, cam_poses, scene_idx, batch_segmap, batch_obj_pcs
        else:
            return batch_data, cam_poses, scene_idx

    def render_random_scene(self, estimate_normals=False, camera_pose=None):
        """
        Renders scene depth map, transforms to regularized pointcloud and applies augmentations

        Keyword Arguments:
            estimate_normals {bool} -- calculate and return normals (default: {False})
            camera_pose {[type]} -- camera pose to render the scene from. (default: {None})

        Returns:
            [pc, pc_normals, camera_pose, depth] -- [point cloud, point cloud normals, camera pose, depth]
        """
        if camera_pose is None:
            viewing_index = np.random.randint(
                0, high=len(self._cam_orientations))
            camera_orientation = self._cam_orientations[viewing_index]
            camera_pose = self.get_cam_pose(camera_orientation)

        in_camera_pose = copy.deepcopy(camera_pose)

        # 0.005 s
        _, depth, _, camera_pose = self._renderer.render(
            in_camera_pose, render_pc=False)
        depth = self._augment_depth(depth)

        pc = self._renderer._to_pointcloud(depth)
        pc = regularize_pc_point_count(
            pc, self._raw_num_points, use_farthest_point=self._use_farthest_point)
        pc = self._augment_pc(pc)

        pc_normals = estimate_normals_cam_from_pc(
            pc[:, :3]) if estimate_normals else []

        return pc, pc_normals, camera_pose, depth

    def change_object(self, cad_path, cad_scale):
        """
        Change object in pyrender scene

        Arguments:
            cad_path {str} -- path to CAD model
            cad_scale {float} -- scale of CAD model
        """

        self._renderer.change_object(cad_path, cad_scale)

    def change_scene(self, obj_paths, obj_scales, obj_transforms, visualize=False):
        """
        Change pyrender scene

        Arguments:
            obj_paths {list[str]} -- path to CAD models in scene
            obj_scales {list[float]} -- scales of CAD models
            obj_transforms {list[np.ndarray]} -- poses of CAD models

        Keyword Arguments:
            visualize {bool} -- whether to update the visualizer as well (default: {False})
        """
        self._renderer.change_scene(obj_paths, obj_scales, obj_transforms)
        if visualize:
            self._visualizer.change_scene(
                obj_paths, obj_scales, obj_transforms)

    def __del__(self):
        print('********** terminating renderer **************')
