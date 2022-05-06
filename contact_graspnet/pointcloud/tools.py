from scipy.spatial import cKDTree
import numpy as np

from ..util.geometry import distance_by_translation_point, farthest_points, inverse_transform, vectorized_normal_computation


def preprocess_pc_for_inference(input_pc, num_point, pc_mean=None, return_mean=False, use_farthest_point=False, convert_to_internal_coords=False):
    """
    Various preprocessing of the point cloud (downsampling, centering, coordinate transforms)  

    Arguments:
        input_pc {np.ndarray} -- Nx3 input point cloud
        num_point {int} -- downsample to this amount of points

    Keyword Arguments:
        pc_mean {np.ndarray} -- use 3x1 pre-computed mean of point cloud  (default: {None})
        return_mean {bool} -- whether to return the point cloud mean (default: {False})
        use_farthest_point {bool} -- use farthest point for downsampling (slow and suspectible to outliers) (default: {False})
        convert_to_internal_coords {bool} -- Convert from opencv to internal coordinates (x left, y up, z front) (default: {False})

    Returns:
        [np.ndarray] -- num_pointx3 preprocessed point cloud
    """
    normalize_pc_count = input_pc.shape[0] != num_point
    if normalize_pc_count:
        pc = regularize_pc_point_count(
            input_pc, num_point, use_farthest_point=use_farthest_point).copy()
    else:
        pc = input_pc.copy()

    if convert_to_internal_coords:
        pc[:, :2] *= -1

    if pc_mean is None:
        pc_mean = np.mean(pc, 0)

    pc -= np.expand_dims(pc_mean, 0)
    if return_mean:
        return pc, pc_mean
    else:
        return pc


def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
      If point cloud pc has less points than npoints, it oversamples.
      Otherwise, it downsample the input pc to have npoint points.
      use_farthest_point: indicates 

      :param pc: Nx3 point cloud
      :param npoints: number of points the regularized point cloud should have
      :param use_farthest_point: use farthest point sampling to downsample the points, runs slower.
      :returns: npointsx3 regularized point cloud
    """

    if pc.shape[0] > npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(
                pc, npoints, distance_by_translation_point, return_center_indexes=True)
        else:
            center_indexes = np.random.choice(
                range(pc.shape[0]), size=npoints, replace=False)
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc


def depth2pc(depth, K, rgb=None):
    """
    Convert depth and intrinsics to point cloud and optionally point cloud color
    :param depth: hxw depth map in m
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: (Nx3 point cloud, point cloud color)
    """

    mask = np.where(depth > 0)
    x, y = mask[1], mask[0]

    normalized_x = (x.astype(np.float32) - K[0, 2])
    normalized_y = (y.astype(np.float32) - K[1, 2])

    world_x = normalized_x * depth[y, x] / K[0, 0]
    world_y = normalized_y * depth[y, x] / K[1, 1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y, x, :]

    pc = np.vstack((world_x, world_y, world_z)).T
    return (pc, rgb)


def estimate_normals_cam_from_pc(pc_cam, max_radius=0.05, k=12):
    """
    Estimates normals in camera coords from given point cloud.

    Arguments:
        pc_cam {np.ndarray} -- Nx3 point cloud in camera coordinates

    Keyword Arguments:
        max_radius {float} -- maximum radius for normal computation (default: {0.05})
        k {int} -- Number of neighbors for normal computation (default: {12})

    Returns:
        [np.ndarray] -- Nx3 point cloud normals
    """
    tree = cKDTree(pc_cam, leafsize=pc_cam.shape[0]+1)
    _, ndx = tree.query(
        pc_cam, k=k, distance_upper_bound=max_radius, n_jobs=-1)  # num_points x k

    for c, idcs in enumerate(ndx):
        idcs[idcs == pc_cam.shape[0]] = c
        ndx[c, :] = idcs
    neighbors = np.array([pc_cam[ndx[:, n], :]
                         for n in range(k)]).transpose((1, 0, 2))
    pc_normals = vectorized_normal_computation(pc_cam, neighbors)
    return pc_normals


def center_pc_convert_cam(cam_poses, batch_data):
    """
    Converts from OpenGL to OpenCV coordinates, computes inverse of camera pose and centers point cloud

    :param cam_poses: (bx4x4) Camera poses in OpenGL format
    :param batch_data: (bxNx3) point clouds 
    :returns: (cam_poses, batch_data) converted
    """
    # OpenCV OpenGL conversion
    for j in range(len(cam_poses)):
        cam_poses[j, :3, 1] = -cam_poses[j, :3, 1]
        cam_poses[j, :3, 2] = -cam_poses[j, :3, 2]
        cam_poses[j] = inverse_transform(cam_poses[j])

    pc_mean = np.mean(batch_data, axis=1, keepdims=True)
    batch_data[:, :, :3] -= pc_mean[:, :, :3]
    cam_poses[:, :3, 3] -= pc_mean[:, 0, :3]

    return cam_poses, batch_data
