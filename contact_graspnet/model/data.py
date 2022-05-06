import glob
import cv2
import numpy as np
from PIL import Image
import os


def load_scene_contacts(dataset_folder, test_split_only=False, num_test=None, scene_contacts_path='scene_contacts_new'):
    """
    Load contact grasp annotations from acronym scenes 

    Arguments:
        dataset_folder {str} -- folder with acronym data and scene contacts

    Keyword Arguments:
        test_split_only {bool} -- whether to only return test split scenes (default: {False})
        num_test {int} -- how many test scenes to use (default: {None})
        scene_contacts_path {str} -- name of folder with scene contact grasp annotations (default: {'scene_contacts_new'})

    Returns:
        list(dicts) -- list of scene annotations dicts with object paths and transforms and grasp contacts and transforms.
    """

    scene_contact_paths = sorted(
        glob.glob(os.path.join(dataset_folder, scene_contacts_path, '*')))
    if test_split_only:
        scene_contact_paths = scene_contact_paths[-num_test:]
    contact_infos = []
    for contact_path in scene_contact_paths:
        print(contact_path)
        try:
            npz = np.load(contact_path, allow_pickle=False)
            contact_info = {'scene_contact_points': npz['scene_contact_points'],
                            'obj_paths': npz['obj_paths'],
                            'obj_transforms': npz['obj_transforms'],
                            'obj_scales': npz['obj_scales'],
                            'grasp_transforms': npz['grasp_transforms']}
            contact_infos.append(contact_info)
        except Exception:
            print('corrupt, ignoring..')
    return contact_infos



def load_available_input_data(p, K=None):
    """
    Load available data from input file path. 

    Numpy files .npz/.npy should have keys
    'depth' + 'K' + (optionally) 'segmap' + (optionally) 'rgb'
    or for point clouds:
    'xyz' + (optionally) 'xyz_color'

    png files with only depth data (in mm) can be also loaded.
    If the image path is from the GraspNet dataset, corresponding rgb, segmap and intrinsic are also loaded.

    :param p: .png/.npz/.npy file path that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: All available data among segmap, rgb, depth, cam_K, pc_full, pc_colors
    """

    segmap, rgb, depth, pc_full, pc_colors = None, None, None, None, None

    if K is not None:
        if isinstance(K, str):
            cam_K = eval(K)
        cam_K = np.array(K).reshape(3, 3)

    if '.np' in p:
        data = np.load(p, allow_pickle=True)
        if '.npz' in p:
            keys = data.files
        else:
            keys = []
            if len(data.shape) == 0:
                data = data.item()
                keys = data.keys()
            elif data.shape[-1] == 3:
                pc_full = data
            else:
                depth = data

        if 'depth' in keys:
            depth = data['depth']
            if K is None and 'K' in keys:
                cam_K = data['K'].reshape(3, 3)
            if 'segmap' in keys:
                segmap = data['segmap']
            if 'seg' in keys:
                segmap = data['seg']
            if 'rgb' in keys:
                rgb = data['rgb']
                rgb = np.array(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        elif 'xyz' in keys:
            pc_full = np.array(data['xyz']).reshape(-1, 3)
            if 'xyz_color' in keys:
                pc_colors = data['xyz_color']
    elif '.png' in p:
        if os.path.exists(p.replace('depth', 'label')):
            # graspnet data
            depth, rgb, segmap, K = load_graspnet_data(p)
        elif os.path.exists(p.replace('depths', 'images').replace('npy', 'png')):
            rgb = np.array(Image.open(
                p.replace('depths', 'images').replace('npy', 'png')))
        else:
            depth = np.array(Image.open(p))
    else:
        raise ValueError('{} is neither png nor npz/npy file'.format(p))

    return segmap, rgb, depth, cam_K, pc_full, pc_colors


def load_graspnet_data(rgb_image_path):
    """
    Loads data from the GraspNet-1Billion dataset
    # https://graspnet.net/

    :param rgb_image_path: .png file path to depth image in graspnet dataset
    :returns: (depth, rgb, segmap, K)
    """

    depth = np.array(Image.open(rgb_image_path))/1000.  # m to mm
    segmap = np.array(Image.open(rgb_image_path.replace('depth', 'label')))
    rgb = np.array(Image.open(rgb_image_path.replace('depth', 'rgb')))

    # graspnet images are upside down, rotate for inference
    # careful: rotate grasp poses back for evaluation
    depth = np.rot90(depth, 2)
    segmap = np.rot90(segmap, 2)
    rgb = np.rot90(rgb, 2)

    if 'kinect' in rgb_image_path:
        # Kinect azure:
        K = np.array([[631.54864502,  0.,     638.43517329],
                      [0.,     631.20751953, 366.49904066],
                      [0.,       0.,       1.]])
    else:
        # Realsense:
        K = np.array([[616.36529541,  0.,     310.25881958],
                      [0.,     616.20294189, 236.59980774],
                      [0.,       0.,       1.]])

    return depth, rgb, segmap, K