import os
import signal

from gorilla.config import Config
import trimesh
import trimesh.transformations as tra
import numpy as np
import cv2
import open3d as o3d

from eval_contactgn_utils.acronym_tools import Scene
from my_eval_oakink_cg import OakinkGraspDataset
from utils import build_dataset
from roboutils.vis.viser_grasp import ViserForGrasp
from roboutils.depth_to_pcd import depth2pc
from roboutils.proj_llm_robot.pose_rep import sevenDof2T, posestamp2T, T2posestamped
from roboutils.proj_llm_robot.pose_transform import update_pose


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
        
        data_dict = {
            "obj_mesh": self.obj_mesh,
            "obj_pose": placement_T,
            "table_mesh": self.table_support,
            "table_pose": np.eye(4),
        }

        # data_dict = [
        #     {"mesh": self.obj_mesh, "pose": placement_T, "name": "obj"},
        #     {"mesh": self.table_support, "pose": np.eye(4), "name": "table_support"},
        #     # {"mesh": self.table_mesh, "pose": self._table_pose, "name": "table"}
        # ]

        return data_dict


import open3d.visualization.gui as gui
class SelectViewGuiApp():
    def __init__(self):
        ToGLCamera = np.array([
            [1,  0,  0,  0],
            [0,  -1,  0,  0],
            [0,  0,  -1,  0],
            [0,  0,  0,  1]
        ])
        self.FromGLGamera = np.linalg.inv(ToGLCamera)

    def model_matrix_to_extrinsic_matrix(self, model_matrix):
        return np.linalg.inv(model_matrix @ self.FromGLGamera)

    def create_camera_intrinsic_from_size(self, width=1024, height=768, hfov=60.0, vfov=60.0):
        fx = (width / 2.0)  / np.tan(np.radians(hfov)/2)
        fy = (height / 2.0)  / np.tan(np.radians(vfov)/2)
        # fx = fy # not sure why, but it looks like fx should be governed/limited by fy
        return np.array(
            [[fx, 0, width / 2.0],
            [0, fy, height / 2.0],
            [0, 0,  1]])

    def save_view(self, vis, fname='saved_view.pkl'):
        try:
            model_matrix = np.asarray(vis.scene.camera.get_model_matrix())
            extrinsic = self.model_matrix_to_extrinsic_matrix(model_matrix)
            width, height = vis.size.width, vis.size.height
            intrinsic = self.create_camera_intrinsic_from_size(width, height)
            saved_view = dict(extrinsic=extrinsic, intrinsic=intrinsic, width=width, height=height)

            # 打印显示当前相机参数
            print("Extrinsic Matrix:\n", np.array2string(np.array(extrinsic), separator=', '))
            print("Intrinsic Matrix:\n", np.array2string(np.array(intrinsic), separator=', '))
            print("Width:", width)
            print("Height:", height)

            # with open(fname, 'wb') as pickle_file:
            #     dump(saved_view, pickle_file)
        except Exception as e:
            print(e)

    # def load_view(self, vis, fname="saved_view.pkl"):
    #     try:
    #         with open(fname, 'rb') as pickle_file:
    #             saved_view = load(pickle_file)
    #         vis.setup_camera(saved_view['intrinsic'], saved_view['extrinsic'], saved_view['width'], saved_view['height'])
    #         # Looks like the ground plane gets messed up, no idea how to fix
    #     except Exception as e:
    #         print("Can't find file", e)
    
    def run(self, geometry_list):
        gui.Application.instance.initialize()
        vis = o3d.visualization.O3DVisualizer("Demo to Load a Camera Viewpoint for O3DVisualizer", 1024, 768)
        gui.Application.instance.add_window(vis)
        vis.point_size = 8
        vis.show_axes = True
        # Add saving and loading view
        vis.add_action("Save Camera View", self.save_view)
        # vis.add_action("Load Camera View", self.load_view)

        for geometry_i, geometry in enumerate(geometry_list):
            vis.add_geometry(f"Geometry {geometry_i}", geometry)

        vis.reset_camera_to_default()
        gui.Application.instance.run()


class Open3dPartialPCDRenderer:
    def __init__(self):
        self.height = 768
        self.width = 1024
        self.K = np.array(
            [[886.81001348,   0.        , 512.        ],
            [  0.        , 665.10751011, 384.        ],
            [  0.        ,   0.        ,   1.        ]]
        )
        self.extrinsic = np.array(
            [[ 9.99285599e-01, -2.90019310e-02,  2.42285147e-02,  6.78054561e-04],
            [-6.86064496e-03, -7.69694573e-01, -6.38375260e-01,  1.52168073e-01],
            [ 3.71626792e-02,  6.37753075e-01, -7.69343790e-01,  7.30794569e-01],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
        )

        pass

    def render_partial_pcd(self, meshes, debug=False):
        device = o3d.core.Device("CPU:0")
        t_meshes = []
        for mesh in meshes:
            t_mesh = o3d.t.geometry.TriangleMesh(device)
            t_mesh.vertex.positions = o3d.core.Tensor(np.asarray(mesh.vertices), o3d.core.float32, device)
            t_mesh.triangle.indices = o3d.core.Tensor(np.asarray(mesh.triangles), o3d.core.int32, device)
            t_meshes.append(t_mesh)
        
        scene = o3d.t.geometry.RaycastingScene()
        object_ids = []
        for mesh in t_meshes:
            obj_id = scene.add_triangles(mesh)
            object_ids.append(obj_id)

        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=self.K,
            extrinsic_matrix=self.extrinsic,
            width_px=int(self.width),
            height_px=int(self.height),
        )
        
        ans = scene.cast_rays(rays)
        depth_image = ans["t_hit"].numpy()
        semantic_image = ans["geometry_ids"].numpy()

        pcd, _, point_idx_to_rgb_crood = depth2pc(depth_image, self.K, max_depth=np.inf)
        pixel_to_point_idx = {tuple(crood): idx for idx, crood in enumerate(point_idx_to_rgb_crood)}
        T = np.linalg.inv(self.extrinsic)
        pcd = (T @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)).T[:, :3]

        unique_obj_ids = np.unique(semantic_image)
        point_mask = np.zeros(pcd.shape[0], dtype=np.uint8)
        for obj_id in unique_obj_ids:
            cur_mask = (semantic_image == obj_id)
            cur_point_idx = np.column_stack(np.where(cur_mask))[:, [1, 0]]
            cur_point_idx = [pixel_to_point_idx[tuple(cur_point_idx[idx])] for idx in range(cur_point_idx.shape[0]) if tuple(cur_point_idx[idx]) in pixel_to_point_idx]
            point_mask[cur_point_idx] = obj_id

        if debug:
            # 验证是否能够对齐
            partial_pcd = o3d.geometry.PointCloud()
            partial_pcd.points = o3d.utility.Vector3dVector(pcd)
            o3d.visualization.draw_geometries([*meshes, partial_pcd])
        
        return {
            "depth_image": depth_image,
            "semantic_obj_ids": object_ids,
            "semantic_image": semantic_image,

            "K": self.K,
            "pcd": pcd,
            "point_semantic_mask": point_mask,
            "pixel_to_point_idx": pixel_to_point_idx,

            "camera_pose": self.extrinsic,
        }


def trimesh_to_o3d(mesh):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d


if __name__ == "__main__":
    from tqdm import tqdm

    work_dir = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_299_20241008-111209_detectiondiffusion"
    config_file_path = os.path.join(work_dir, "config.py")
    cfg = Config.fromfile(config_file_path)
    dataset = build_dataset(cfg)['test_set']

    oakink_data_root = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/data/grasp_Oakink"
    oakink_dataset = OakinkGraspDataset(oakink_data_root, dataset)

    # # @note ################### 1. 获得单个物体的局部观测点云并缓存下来
    # save_dir = os.path.join(oakink_data_root, "scene_partial_data")
    # os.makedirs(save_dir, exist_ok=True)
    # select_view_gui_app = SelectViewGuiApp()
    # scene = OurOakinkTableScene()
    # render_partial_pcd = Open3dPartialPCDRenderer()
    # viser_grasp = ViserForGrasp()
    # for data_i, data in tqdm(enumerate(oakink_dataset), total=len(oakink_dataset)):
    #     if data_i < 160:
    #         continue

    #     data_path, mesh_path, obj_verts, mesh_T, my_palm_T, hand_verts = data
    #     data_path, mesh_path = data_path[0], mesh_path[0]
    #     data_base_name = os.path.basename(data_path)

    #     obj_mesh = trimesh.load(mesh_path)

    #     scene.set_current_obj_mesh(obj_mesh)
    #     data_dict = scene.arrange(max_iter=100, time_out=8)

    #     obj_pose = data_dict["obj_pose"]
    #     table_mesh = data_dict["table_mesh"]
    #     table_pose = data_dict["table_pose"]

    #     o3d_obj_mesh = o3d.io.read_triangle_mesh(mesh_path)
    #     o3d_table_mesh = trimesh_to_o3d(table_mesh)
    #     o3d_obj_mesh.transform(obj_pose)
    #     o3d_table_mesh.transform(table_pose)

    #     # @note 选择相机参数
    #     # select_view_gui_app.run([o3d_obj_mesh, o3d_table_mesh])

    #     # @note 获得单物体放到场景的场景局部观测点云
    #     debug = False
    #     render_data = render_partial_pcd.render_partial_pcd([o3d_obj_mesh, o3d_table_mesh], debug=debug)

    #     depth_image = render_data["depth_image"]
    #     object_ids = render_data["semantic_obj_ids"]
    #     semantic_image = render_data["semantic_image"]

    #     camera_K = render_data["K"]
    #     camera_pose = render_data["camera_pose"]

    #     pcd = render_data["pcd"]
    #     pixel_to_point_idx = render_data["pixel_to_point_idx"]
    #     point_semantic_mask = render_data["point_semantic_mask"]

    #     render_data["obj_mesh_pose"] = obj_pose

    #     # # debug
    #     # cv2.imshow("obj_mask", (semantic_image == object_ids[0]).astype(np.uint8) * 255)
    #     # cv2.imshow("table_mask", (semantic_image == object_ids[1]).astype(np.uint8) * 255)
    #     # cv2.waitKey(0)
    #     # # debug
    #     # obj_pcd = pcd[point_semantic_mask == object_ids[0]]
    #     # table_pcd = pcd[point_semantic_mask == object_ids[1]]
    #     # vis_obj_pcd = o3d.geometry.PointCloud()
    #     # vis_obj_pcd.points = o3d.utility.Vector3dVector(obj_pcd)
    #     # vis_obj_pcd.paint_uniform_color([1, 0, 0])
    #     # vis_table_pcd = o3d.geometry.PointCloud()
    #     # vis_table_pcd.points = o3d.utility.Vector3dVector(table_pcd)
    #     # o3d.visualization.draw_geometries([vis_obj_pcd, vis_table_pcd])

    #     np.save(os.path.join(save_dir, str(data_i) + ".npy"), render_data)
        
    #     scene.reset()
    #     pass
    # print("保存第一步结果成功！")

    # @note ################### 2. 使用 contact-graspnet 生成抓取
    import rospy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped, Pose
    from sensor_msgs.msg import Image, CameraInfo
    from std_msgs.msg import (Float64, Float64MultiArray, MultiArrayDimension,
                          MultiArrayLayout, String)
    from interfaces.srv import ControlPose, ControlJoints, PredictGrasp, SegmentImage
    from interfaces.srv import ControlPoseRequest, ControlJointsRequest, PredictGraspRequest, SegmentImageRequest

    rospy.wait_for_service('pred_grasp', timeout=2.0)
    graspnet_client = rospy.ServiceProxy('pred_grasp', PredictGrasp)
    cv_bridge = CvBridge()

    save_dir = os.path.join(oakink_data_root, "scene_partial_data")
    os.makedirs(save_dir, exist_ok=True)
    viser_grasp = ViserForGrasp()
    results = []
    for data_i, data in tqdm(enumerate(oakink_dataset), total=len(oakink_dataset)):
        data_path, mesh_path, obj_verts, mesh_T, my_palm_T, hand_verts = data
        data_path, mesh_path = data_path[0], mesh_path[0]
        data_base_name = os.path.basename(data_path)

        data_i_path = os.path.join(save_dir, str(data_i) + ".npy")
        render_data = np.load(data_i_path, allow_pickle=True).item()
        obj_mesh_pose = render_data["obj_mesh_pose"]

        depth_image = render_data["depth_image"]
        object_ids = render_data["semantic_obj_ids"]
        semantic_image = render_data["semantic_image"]

        camera_K = render_data["K"]
        camera_pose = render_data["camera_pose"]

        pcd = render_data["pcd"]
        pixel_to_point_idx = render_data["pixel_to_point_idx"]
        point_semantic_mask = render_data["point_semantic_mask"]

        # 调用 contact-graspnet 服务
        fake_rgb_image = np.zeros((768, 1024, 3), dtype=np.uint8)
        object_mask = (semantic_image == object_ids[0]).astype(np.uint8)
        rgb_msg = cv_bridge.cv2_to_imgmsg(fake_rgb_image, encoding='bgr8')
        depth_image = (depth_image * 1000.0).astype(np.uint16)
        depth_msg = cv_bridge.cv2_to_imgmsg(depth_image, encoding='16UC1')
        mask_msg = cv_bridge.cv2_to_imgmsg(object_mask.astype(np.uint8), encoding='mono8')
        req = PredictGraspRequest()
        req.rgb = rgb_msg
        req.depth = depth_msg
        req.intrinsic = Float64MultiArray(data=camera_K.flatten())
        req.mask = mask_msg

        repeat_time = 10
        best_grasp_Ts = []
        for i in range(repeat_time):
            reponse = graspnet_client(req)
            grasp_poses = reponse.grasp_poses
            grasp_confidences = reponse.grasp_confidences
            grasp_Ts = np.array([posestamp2T(pose) for pose in grasp_poses])
            grasp_confidences = np.array(grasp_confidences.data)
            grasp_Ts = grasp_Ts[np.argsort(grasp_confidences)[::-1]]
            if len(grasp_Ts) == 0:
                continue
            best_grasp_Ts.append(grasp_Ts[0])
        best_grasp_Ts = np.array(best_grasp_Ts)
        grasp_Ts = best_grasp_Ts
        
        if len(grasp_Ts) == 0:
            print("No grasp found!")
            grasp_Ts = np.array([np.eye(4)] * 10)
        while len(grasp_Ts) < 10:
            grasp_Ts = np.concatenate([grasp_Ts, np.array([np.eye(4)] * (10 - len(grasp_Ts)))], axis=0)

        tmp_grasp_Ts = []
        for grasp_T in grasp_Ts:
            grasp_T = update_pose(grasp_T, rotate=-np.pi / 2, rotate_axis='x')
            grasp_T = update_pose(grasp_T, rotate=np.pi / 2, rotate_axis='y')
            grasp_T = update_pose(grasp_T, translate=[0, 0, -0.08])
            tmp_grasp_Ts.append(grasp_T)
        grasp_Ts = tmp_grasp_Ts

        world_grasp_Ts = [np.linalg.inv(camera_pose) @ grasp_T for grasp_T in grasp_Ts]
        # # debug
        # viser_grasp.vis_grasp_scene(world_grasp_Ts, pcd, z_direction=True)
        # viser_grasp.wait_for_reset()

        # 转换为物体 mesh 坐标系
        obj_mesh_grasp_Ts = [np.linalg.inv(obj_mesh_pose) @ grasp_T for grasp_T in world_grasp_Ts]
        # # debug
        # o3d_obj_mesh = o3d.io.read_triangle_mesh(mesh_path)
        # viser_grasp.vis_grasp_scene(obj_mesh_grasp_Ts, mesh=o3d_obj_mesh, z_direction=True)
        # viser_grasp.wait_for_reset()

        data_dict = {
            'mesh_path': mesh_path,
            'xyz': pcd,
            'grasp_Ts': obj_mesh_grasp_Ts[:10],  # 选择前 10 个抓取姿态
            'mesh_T': np.eye(4),
        }
        results.append(data_dict)

    # @note 导出与人手抓取无关的抓取数据评估
    # 将结果转换为 IsaacGym 格式
    save_name = "contactgn_oakink_best"
    isaacgym_eval_data_dict = {}
    for i, data_dict in tqdm(enumerate(results), total=len(results)):
        isaacgym_eval_data_dict[i] = {}

        obj_pc = data_dict['xyz']
        mesh_T = data_dict['mesh_T']
        mesh_path = data_dict['mesh_path']
        mesh_scale = 1.0

        isaacgym_eval_data_dict[i]['mesh_path'] = mesh_path
        isaacgym_eval_data_dict[i]['mesh_scale'] = mesh_scale
        isaacgym_eval_data_dict[i]['grasp_Ts'] = data_dict['grasp_Ts']
        isaacgym_eval_data_dict[i]['mesh_T'] = mesh_T

        # # for debug vis
        # mesh = o3d.io.read_triangle_mesh(mesh_path)
        # mesh.scale(mesh_scale, center=np.zeros(3))
        # mesh.transform(mesh_T)
        # grasp_Ts = data_dict['grasp_Ts']
        # viser_for_grasp.vis_grasp_scene(grasp_Ts, mesh=mesh, pc=obj_pc, max_grasp_num=50)
        # viser_for_grasp.wait_for_reset()

    np.save(os.path.join(work_dir, '{}_isaacgym.npy'.format(save_name)), isaacgym_eval_data_dict)
