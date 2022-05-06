
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mayavi.mlab as mlab
import tensorflow.compat.v1 as tf
import trimesh


class Gripper(object):

    def __init__(self, q=None):
        self.base = None
        self.fingers = []
        self.contact_ray_origins = []
        self.contact_ray_directions = []
        self.control_points = []
        self.q = q

    @property
    def hand(self):
        return trimesh.util.concatenate([*self.fingers, self.base])

    @property
    def all_fingers(self):
        return trimesh.util.concatenate(self.fingers)

    def get_meshes(self):
        """Get list of meshes that this gripper consists of.

        Returns:
            list of trimesh -- visual meshes
        """
        return [*self.fingers, self.base]

    def get_closing_rays_contact(self, transform):
        """Get an array of rays defining the contact locations and directions on the hand.

        Arguments:
            transform {[numpy.array]} -- a 4x4 homogeneous matrix
            contact_ray_origin {[numpy.array]} -- a 4x1 homogeneous vector
            contact_ray_direction {[numpy.array]} -- a 4x1 homogeneous vector

        Returns:
            numpy.array -- transformed rays (origin and direction)
        """
        return transform[:3, :].dot(
            self.contact_ray_origins.T).T, transform[:3, :3].dot(self.contact_ray_directions.T).T

    def get_control_point_tensor(self, batch_size, use_tf=True, symmetric=False, convex_hull=True):
        """
        Outputs a 5 point gripper representation of shape (batch_size x 5 x 3).

        Arguments:
            batch_size {int} -- batch size

        Keyword Arguments:
            use_tf {bool} -- outputing a tf tensor instead of a numpy array (default: {True})
            symmetric {bool} -- Output the symmetric control point configuration of the gripper (default: {False})
            convex_hull {bool} -- Return control points according to the convex hull panda gripper model (default: {True})

        Returns:
            np.ndarray -- control points of the panda gripper 
        """
        if symmetric:
            control_points = [[0, 0, 0], self.control_points[1, :],
                              self.control_points[0, :], self.control_points[-1, :], self.control_points[-2, :]]
        else:
            control_points = [[0, 0, 0], self.control_points[0, :],
                              self.control_points[1, :], self.control_points[-2, :], self.control_points[-1, :]]

        control_points = np.asarray(control_points, dtype=np.float32)
        if not convex_hull:
            # actual depth of the gripper different from convex collision model
            control_points[1:3, 2] = 0.0584
        control_points = np.tile(np.expand_dims(
            control_points, 0), [batch_size, 1, 1])

        if use_tf:
            return tf.convert_to_tensor(control_points)

        return control_points

    def plot_gripper(self, cam_trafo=np.eye(4), mesh_pose=np.eye(4)):
        """
        Plots mesh in mesh_pose from 

        Arguments:
            mesh {trimesh.base.Trimesh} -- input mesh, e.g. gripper

        Keyword Arguments:
            cam_trafo {np.ndarray} -- 4x4 transformation from world to camera coords (default: {np.eye(4)})
            mesh_pose {np.ndarray} -- 4x4 transformation from mesh to world coords (default: {np.eye(4)})
        """

        homog_mesh_vert = np.pad(self.hand.vertices, (0, 1),
                                 'constant', constant_values=(0, 1))
        mesh_cam = homog_mesh_vert.dot(mesh_pose.T).dot(cam_trafo.T)[:, :3]
        mlab.triangular_mesh(mesh_cam[:, 0],
                             mesh_cam[:, 1],
                             mesh_cam[:, 2],
                             self.hand.faces,
                             colormap='Blues',
                             opacity=0.5)

    def draw_grasps(self, grasps, cam_pose, gripper_openings, color=(0, 1., 0), colors=None, show_gripper_mesh=False, tube_radius=0.0008):
        """
        Draws wireframe grasps from given camera pose and with given gripper openings

        Arguments:
            grasps {np.ndarray} -- Nx4x4 grasp pose transformations
            cam_pose {np.ndarray} -- 4x4 camera pose transformation
            gripper_openings {np.ndarray} -- Nx1 gripper openings

        Keyword Arguments:
            color {tuple} -- color of all grasps (default: {(0,1.,0)})
            colors {np.ndarray} -- Nx3 color of each grasp (default: {None})
            tube_radius {float} -- Radius of the grasp wireframes (default: {0.0008})
            show_gripper_mesh {bool} -- Renders the gripper mesh for one of the grasp poses (default: {False})
        """

        gripper_control_points = self.get_control_point_tensor(
            1, False, convex_hull=False).squeeze()
        mid_point = 0.5 * \
            (gripper_control_points[1, :] + gripper_control_points[2, :])
        grasp_line_plot = np.array([np.zeros((3,)), mid_point, gripper_control_points[1], gripper_control_points[3],
                                    gripper_control_points[1], gripper_control_points[2], gripper_control_points[4]])

        if show_gripper_mesh and len(grasps) > 0:
            self.plot_gripper(cam_pose, grasps[0])

        all_pts = []
        connections = []
        index = 0
        N = 7
        for i, (g, g_opening) in enumerate(zip(grasps, gripper_openings)):
            gripper_control_points_closed = grasp_line_plot.copy()
            gripper_control_points_closed[2:, 0] = np.sign(
                grasp_line_plot[2:, 0]) * g_opening/2

            pts = np.matmul(gripper_control_points_closed, g[:3, :3].T)
            pts += np.expand_dims(g[:3, 3], 0)
            pts_homog = np.concatenate((pts, np.ones((7, 1))), axis=1)
            pts = np.dot(pts_homog, cam_pose.T)[:, :3]

            color = color if colors is None else colors[i]

            all_pts.append(pts)
            connections.append(np.vstack([np.arange(index,   index + N - 1.5),
                                          np.arange(index + 1, index + N - .5)]).T)
            index += N
            # mlab.plot3d(pts[:, 0], pts[:, 1], pts[:, 2], color=color, tube_radius=tube_radius, opacity=1.0)

        # speeds up plot3d because only one vtk object
        all_pts = np.vstack(all_pts)
        connections = np.vstack(connections)
        src = mlab.pipeline.scalar_scatter(
            all_pts[:, 0], all_pts[:, 1], all_pts[:, 2])
        src.mlab_source.dataset.lines = connections
        src.update()
        lines = mlab.pipeline.tube(src, tube_radius=tube_radius, tube_sides=12)
        mlab.pipeline.surface(lines, color=color, opacity=1.0)

    def visualize_grasps(self, full_pc, pred_grasps_cam, scores, plot_opencv_cam=False, pc_colors=None, gripper_openings=None, gripper_width=0.08):
        """Visualizes colored point cloud and predicted grasps. If given, colors grasps by segmap regions. 
        Thick grasp is most confident per segment. For scene point cloud predictions, colors grasps according to confidence.

        Arguments:
            full_pc {np.ndarray} -- Nx3 point cloud of the scene
            pred_grasps_cam {dict[int:np.ndarray]} -- Predicted 4x4 grasp trafos per segment or for whole point cloud
            scores {dict[int:np.ndarray]} -- Confidence scores for grasps

        Keyword Arguments:
            plot_opencv_cam {bool} -- plot camera coordinate frame (default: {False})
            pc_colors {np.ndarray} -- Nx3 point cloud colors (default: {None})
            gripper_openings {dict[int:np.ndarray]} -- Predicted grasp widths (default: {None})
            gripper_width {float} -- If gripper_openings is None, plot grasp widths (default: {0.008})
        """

        print('Visualizing...takes time')
        cm = plt.get_cmap('rainbow')
        cm2 = plt.get_cmap('gist_rainbow')

        # fig = mlab.figure('Pred Grasps')
        mlab.view(azimuth=180, elevation=180, distance=0.2)
        draw_pc_with_colors(full_pc, pc_colors)
        colors = [cm(1. * i/len(pred_grasps_cam))[:3]
                  for i in range(len(pred_grasps_cam))]
        colors2 = {k: cm2(0.5*np.max(scores[k]))[:3]
                   for k in pred_grasps_cam if np.any(pred_grasps_cam[k])}

        if plot_opencv_cam:
            plot_coordinates(np.zeros(3,), np.eye(3, 3))
        for i, k in enumerate(pred_grasps_cam):
            if np.any(pred_grasps_cam[k]):
                gripper_openings_k = np.ones(len(
                    pred_grasps_cam[k]))*gripper_width if gripper_openings is None else gripper_openings[k]
                if len(pred_grasps_cam) > 1:
                    self.draw_grasps(pred_grasps_cam[k], np.eye(
                        4), color=colors[i], gripper_openings=gripper_openings_k)
                    self.draw_grasps([pred_grasps_cam[k][np.argmax(scores[k])]], np.eye(4), color=colors2[k],
                                     gripper_openings=[gripper_openings_k[np.argmax(scores[k])]], tube_radius=0.0025)
                else:
                    colors3 = [cm2(0.5*score)[:3] for score in scores[k]]
                    self.draw_grasps(pred_grasps_cam[k], np.eye(
                        4), colors=colors3, gripper_openings=gripper_openings_k)
        mlab.show()

    def in_collision_with_gripper(self, object_mesh, gripper_transforms, silent=False):
        """Check collision of object with gripper.

        Arguments:
            object_mesh {trimesh} -- mesh of object
            gripper_transforms {list of numpy.array} -- homogeneous matrices of gripper

        Keyword Arguments:
            silent {bool} -- verbosity (default: {False})

        Returns:
            [list of bool] -- Which gripper poses are in collision with object mesh
        """
        manager = trimesh.collision.CollisionManager()
        manager.add_object('object', object_mesh)
        gripper_meshes = [self.hand]
        min_distance = []
        for tf in tqdm(gripper_transforms, disable=silent):
            min_distance.append(np.min([manager.min_distance_single(
                gripper_mesh, transform=tf) for gripper_mesh in gripper_meshes]))

        return [d == 0 for d in min_distance], min_distance

    def grasp_contact_location(self, transforms, successfulls, collisions, object_mesh, silent=False):
        """Computes grasp contacts on objects and normals, offsets, directions

        Arguments:
            transforms {[type]} -- grasp poses
            collisions {[type]} -- collision information
            object_mesh {trimesh} -- object mesh

        Keyword Arguments:
            silent {bool} -- verbosity (default: {False})

        Returns:
            list of dicts of contact information per grasp ray
        """
        res = []
        if trimesh.ray.has_embree:
            intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
                object_mesh, scale_to_box=True)
        else:
            intersector = trimesh.ray.ray_triangle.RayMeshIntersector(
                object_mesh)
        for p, colliding, outcome in tqdm(zip(transforms, collisions, successfulls), total=len(transforms), disable=silent):
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
                ray_origins, ray_directions = self.get_closing_rays_contact(p)

                locations, index_ray, index_tri = intersector.intersects_location(
                    ray_origins, ray_directions, multiple_hits=False)

                if len(locations) > 0:
                    # this depends on the width of the gripper
                    valid_locations = np.linalg.norm(
                        ray_origins[index_ray]-locations, axis=1) <= 2.0*self.q

                    if sum(valid_locations) > 1:
                        contact_dict['valid_locations'] = 1
                        contact_dict['contact_points'] = locations[valid_locations]
                        contact_dict['contact_face_normals'] = object_mesh.face_normals[index_tri[valid_locations]]
                        contact_dict['contact_directions'] = ray_directions[index_ray[valid_locations]]
                        contact_dict['contact_offsets'] = np.linalg.norm(
                            ray_origins[index_ray[valid_locations]] - locations[valid_locations], axis=1)
                        # dot_prods = (contact_dict['contact_face_normals'] * contact_dict['contact_directions']).sum(axis=1)
                        # contact_dict['contact_cosine_angles'] = np.cos(dot_prods)
                        res.append(contact_dict)

        return res


def plot_coordinates(t, r, tube_radius=0.005):
    """
    plots coordinate frame

    Arguments:
        t {np.ndarray} -- translation vector
        r {np.ndarray} -- rotation matrix

    Keyword Arguments:
        tube_radius {float} -- radius of the plotted tubes (default: {0.005})
    """
    mlab.plot3d([t[0], t[0]+0.2*r[0, 0]], [t[1], t[1]+0.2*r[1, 0]], [t[2],
                t[2]+0.2*r[2, 0]], color=(1, 0, 0), tube_radius=tube_radius, opacity=1)
    mlab.plot3d([t[0], t[0]+0.2*r[0, 1]], [t[1], t[1]+0.2*r[1, 1]], [t[2],
                t[2]+0.2*r[2, 1]], color=(0, 1, 0), tube_radius=tube_radius, opacity=1)
    mlab.plot3d([t[0], t[0]+0.2*r[0, 2]], [t[1], t[1]+0.2*r[1, 2]], [t[2],
                t[2]+0.2*r[2, 2]], color=(0, 0, 1), tube_radius=tube_radius, opacity=1)


def draw_pc_with_colors(pc, pc_colors=None, single_color=(0.3, 0.3, 0.3), mode='2dsquare', scale_factor=0.0018):
    """
    Draws colored point clouds

    Arguments:
        pc {np.ndarray} -- Nx3 point cloud
        pc_colors {np.ndarray} -- Nx3 point cloud colors

    Keyword Arguments:
        single_color {tuple} -- single color for point cloud (default: {(0.3,0.3,0.3)})
        mode {str} -- primitive type to plot (default: {'point'})
        scale_factor {float} -- Scale of primitives. Does not work for points. (default: {0.002})

    """

    if pc_colors is None:
        mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2],
                      color=single_color, scale_factor=scale_factor, mode=mode)
    else:
        # create direct grid as 256**3 x 4 array
        def create_8bit_rgb_lut():
            xl = np.mgrid[0:256, 0:256, 0:256]
            lut = np.vstack((xl[0].reshape(1, 256**3),
                             xl[1].reshape(1, 256**3),
                             xl[2].reshape(1, 256**3),
                             255 * np.ones((1, 256**3)))).T
            return lut.astype('int32')

        scalars = pc_colors[:, 0]*256**2 + \
            pc_colors[:, 1]*256 + pc_colors[:, 2]
        rgb_lut = create_8bit_rgb_lut()
        points_mlab = mlab.points3d(
            pc[:, 0], pc[:, 1], pc[:, 2], scalars, mode=mode, scale_factor=.0018)
        points_mlab.glyph.scale_mode = 'scale_by_vector'
        points_mlab.module_manager.scalar_lut_manager.lut._vtk_obj.SetTableRange(
            0, rgb_lut.shape[0])
        points_mlab.module_manager.scalar_lut_manager.lut.number_of_colors = rgb_lut.shape[0]
        points_mlab.module_manager.scalar_lut_manager.lut.table = rgb_lut
