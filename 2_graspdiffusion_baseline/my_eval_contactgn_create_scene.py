import os
import signal

from gorilla.config import Config
import trimesh
import trimesh.transformations as tra
import numpy as np
import open3d as o3d

from eval_contactgn_utils.acronym_tools import Scene
from my_eval_oakink_cg import OakinkGraspDataset
from utils import build_dataset
from roboutils.vis.viser_grasp import ViserForGrasp


class OurOakinkTableScene(Scene):
    def __init__(self):
        super().__init__()

        self.obj_mesh = None

        self._table_dims = [1.0, 1.2, 0.6]
        self._table_support = [0.6, 0.6, 0.6]
        self._table_pose = np.eye(4)
        self.table_mesh = trimesh.creation.box(self._table_dims)
        self.table_support = trimesh.creation.box(self._table_support)
        self._lower_table = 0.02
        pass

    def set_current_obj_mesh(self, obj_mesh):
        self.obj_mesh = obj_mesh
        pass

    def reset(self):
        """
        Reset, i.e. remove scene objects
        """
        for name in self._objects:
            self.collision_manager.remove_object(name)
        self._objects = {}
        self._poses = {}
        self._support_objects = []

    def is_colliding(self, mesh, transform, eps=1e-6):
        """
        Whether given mesh collides with scene

        Arguments:
            mesh {trimesh.Trimesh} -- mesh 
            transform {np.ndarray} -- mesh transform

        Keyword Arguments:
            eps {float} -- minimum distance detected as collision (default: {1e-6})

        Returns:
            [bool] -- colliding or not
        """
        dist = self.collision_manager.min_distance_single(mesh, transform=transform)
        return dist < eps

    def find_object_placement(self, obj_mesh, max_iter):
        """Try to find a non-colliding stable pose on top of any support surface.

        Args:
            obj_mesh (trimesh.Trimesh): Object mesh to be placed.
            max_iter (int): Maximum number of attempts to place to object randomly.

        Raises:
            RuntimeError: In case the support object(s) do not provide any support surfaces.

        Returns:
            bool: Whether a placement pose was found.
            np.ndarray: Homogenous 4x4 matrix describing the object placement pose. Or None if none was found.
        """
        support_polys, support_T = self._get_support_polygons()
        if len(support_polys) == 0:
            raise RuntimeError("No support polygons found!")

        # get stable poses for object
        stable_obj = obj_mesh.copy()
        stable_obj.vertices -= stable_obj.center_mass
        stable_poses, stable_poses_probs = stable_obj.compute_stable_poses(
            threshold=0, sigma=0, n_samples=20
        )

        # Sample support index
        support_index = max(enumerate(support_polys), key=lambda x: x[1].area)[0]

        iter = 0
        colliding = True
        while iter < max_iter and colliding:

            # Sample position in plane
            pts = trimesh.path.polygons.sample(
                support_polys[support_index], count=1
            )

            # To avoid collisions with the support surface
            pts3d = np.append(pts, 0)

            # Transform plane coordinates into scene coordinates
            placement_T = np.dot(
                support_T[support_index],
                trimesh.transformations.translation_matrix(pts3d),
            )

            pose = self._get_random_stable_pose(stable_poses, stable_poses_probs)

            placement_T = np.dot(
                np.dot(placement_T, pose), tra.translation_matrix(-obj_mesh.center_mass)
            )

            # 放到桌面中心
            placement_T[0, 3] = 0.0
            placement_T[1, 3] = 0.0

            # Check collisions
            colliding = self.is_colliding(obj_mesh, placement_T)

            iter += 1

        return not colliding, placement_T if not colliding else None
    
    def find_not_stable_placement(self, obj_mesh):
        # 不断调高，直到与桌子不发生碰撞即可
        placement_T = np.eye(4)
        placement_T[2, 3] = 0.1

        while True:
            # Check collisions
            tmp_placement_T = np.dot(placement_T, tra.translation_matrix(obj_mesh.center_mass))
            colliding = self.is_colliding(obj_mesh, tmp_placement_T)
            if not colliding:
                placement_T = tmp_placement_T
                break
            placement_T[2, 3] += 0.01
        return placement_T

    def handler(self, signum, frame):
        raise Exception("Could not place object ")

    def arrange(self, max_iter=100, time_out = 8):
        self._table_pose[2,3] -= self._lower_table
        self.add_object('table', self.table_mesh, self._table_pose)       
        self._support_objects.append(self.table_support) 

        signal.signal(signal.SIGALRM, self.handler)
        signal.alarm(time_out)
        try:
            success, placement_T = self.find_object_placement(self.obj_mesh, max_iter)
            signal.alarm(0)
        except Exception as exc: 
            print(exc, "Timeout after {} seconds!".format(time_out))
            self.add_object('table_support', self.table_support, np.eye(4))
            placement_T = self.find_not_stable_placement(self.obj_mesh)

        data_dict = [
            {"mesh": self.obj_mesh, "pose": placement_T, "name": "obj"},
            {"mesh": self.table_support, "pose": np.eye(4), "name": "table_support"},
            # {"mesh": self.table_mesh, "pose": self._table_pose, "name": "table"}
        ]
        return data_dict


def trimesh_to_o3d(mesh):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d


if __name__ == "__main__":
    work_dir = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_299_20241008-111209_detectiondiffusion"
    config_file_path = os.path.join(work_dir, "config.py")
    cfg = Config.fromfile(config_file_path)
    dataset = build_dataset(cfg)['test_set']

    oakink_data_root = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/data/grasp_Oakink"
    oakink_dataset = OakinkGraspDataset(oakink_data_root, dataset)

    scene = OurOakinkTableScene()
    viser_grasp = ViserForGrasp()
    for data_i, data in enumerate(oakink_dataset):
        data_path, mesh_path, obj_verts, mesh_T, my_palm_T, hand_verts = data
        data_path, mesh_path = data_path[0], mesh_path[0]

        obj_mesh = trimesh.load(mesh_path)

        scene.set_current_obj_mesh(obj_mesh)
        data_dict = scene.arrange(max_iter=100, time_out=8)

        for data_dict_i in data_dict:
            mesh = data_dict_i["mesh"]
            pose = data_dict_i["pose"]
            name = data_dict_i["name"]

            o3d_mesh = trimesh_to_o3d(mesh)
            o3d_mesh.transform(pose)

            viser_grasp.add_mesh(o3d_mesh, name=name)
        viser_grasp.wait_for_reset()

        scene.reset()

        pass


