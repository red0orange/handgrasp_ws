import torch
import numpy as np
import open3d as o3d

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    look_at_view_transform,
    MeshRasterizer,
    RasterizationSettings,
    TexturesVertex,
)
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix
from pytorch3d.io import load_obj

from roboutils.depth_to_pcd import depth2pc


class Pytorch3dPartialPCDRenderer:
    def __init__(self, render_distance=3.0):
        self.height = 512
        self.width = 512
        self.fx = 512
        self.fy = 512
        self.cx = self.width / 2
        self.cy = self.height / 2
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.render_distance = render_distance
        self.render_R = torch.eye(3).unsqueeze(0)
        self.render_t = torch.tensor([0.0, 0.0, self.render_distance]).unsqueeze(0)

        self.render_T = np.eye(4)
        self.render_T[:3, :3] = self.render_R[0].cpu().numpy()
        self.render_T[:3, 3] = self.render_t[0].cpu().numpy()

        self.cameras = PerspectiveCameras(
            image_size=[[self.width, self.height]],
            R=self.render_R[None],
            T=self.render_t[None],
            focal_length=torch.tensor([[self.fx, self.fy]], dtype=torch.float32),
            principal_point=torch.tensor([[self.cx, self.cy]], dtype=torch.float32),
            in_ndc=False,
        )

        self.rasterizer = MeshRasterizer(
            cameras=self.cameras,
            raster_settings=RasterizationSettings(
                image_size=[self.height, self.width],
            ),
        )
        pass

    def render_partial_pcd(self, mesh_verts, mesh_faces, render_rot_R, debug=False):
        ori_mesh_verts = mesh_verts.copy()
        mesh_centriod = mesh_verts.mean(0)
        mesh_verts = mesh_verts - mesh_centriod[None, ...]

        render_rot_T = np.eye(4)
        render_rot_T[:3, :3] = render_rot_R
        mesh_verts = (render_rot_T @ np.concatenate([mesh_verts.T, np.ones((1, mesh_verts.shape[0]))], axis=0)).T[:, :3]

        mesh_verts = torch.tensor(mesh_verts, dtype=torch.float32)
        mesh_faces = torch.tensor(mesh_faces, dtype=torch.long)
        mesh = Meshes(
            verts=[mesh_verts],
            faces=[mesh_faces],
            textures=TexturesVertex(verts_features=torch.ones_like(mesh_verts)[None]),
        )
        
        # render
        fragments = self.rasterizer(meshes_world=mesh, R=self.render_R, T=self.render_t)
        depth_image = fragments.zbuf[0, ..., 0].cpu().numpy()

        pcd, _, _ = depth2pc(depth_image, self.K, max_depth=np.inf)
        T = np.linalg.inv(self.render_T)
        axis_T = np.eye(4)
        axis_T[:3, :3] = euler_angles_to_matrix(torch.tensor([0.0, 0.0, np.pi]), "XYZ").numpy()
        pcd = (np.linalg.inv(render_rot_T) @ axis_T @ T @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)).T[:, :3]
        pcd = pcd + mesh_centriod[None, ...]

        if debug:
            # @debug
            ori_mesh = o3d.geometry.TriangleMesh()
            ori_mesh.vertices = o3d.utility.Vector3dVector(ori_mesh_verts)
            ori_mesh.triangles = o3d.utility.Vector3iVector(mesh_faces)
            ori_mesh.compute_vertex_normals()
            partial_pcd = o3d.geometry.PointCloud()
            partial_pcd.points = o3d.utility.Vector3dVector(pcd)
            o3d.visualization.draw_geometries([ori_mesh, partial_pcd])
        
        return pcd


class Open3dPartialPCDRenderer:
    def __init__(self):
        self.height = 400
        self.width = 400
        self.fx = 348.49858714
        self.fy = 348.49858714
        self.cx = self.width / 2
        self.cy = self.height / 2
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.render_R = np.eye(3)
        self.render_T = np.eye(4)
        self.render_T[:3, :3] = self.render_R
        pass

    def render_partial_pcd(self, mesh_verts, mesh_faces, render_rot_R, debug=False):
        ori_mesh_verts = mesh_verts.copy()
        mesh_centriod = mesh_verts.mean(0)
        mesh_verts = mesh_verts - mesh_centriod[None, ...]

        render_rot_T = np.eye(4)
        render_rot_T[:3, :3] = render_rot_R
        mesh_verts = (render_rot_T @ np.concatenate([mesh_verts.T, np.ones((1, mesh_verts.shape[0]))], axis=0)).T[:, :3]

        # 根据物体大小设置相机距离
        # distance = np.max(mesh_verts, axis=0)[2] * 20
        distance = 2
        self.render_T[:3, 3] = np.array([0.0, 0.0, distance])

        device = o3d.core.Device("CPU:0")
        dtype_f = o3d.core.float32
        dtype_i = o3d.core.int32
        mesh = o3d.t.geometry.TriangleMesh(device)
        mesh.vertex.positions = o3d.core.Tensor(mesh_verts, dtype_f, device)
        mesh.triangle.indices = o3d.core.Tensor(mesh_faces, dtype_i, device)

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)

        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=self.K,
            extrinsic_matrix=self.render_T,
            width_px=int(self.width),
            height_px=int(self.height),
        )
        ans = scene.cast_rays(rays)
        depth_image = ans["t_hit"].numpy()

        pcd, _, _ = depth2pc(depth_image, self.K, max_depth=np.inf)
        T = np.linalg.inv(self.render_T)
        pcd = (np.linalg.inv(render_rot_T) @ T @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)).T[:, :3]
        pcd = pcd + mesh_centriod[None, ...]

        if debug:
            # @debug
            ori_mesh = o3d.geometry.TriangleMesh()
            ori_mesh.vertices = o3d.utility.Vector3dVector(ori_mesh_verts)
            ori_mesh.triangles = o3d.utility.Vector3iVector(mesh_faces)
            ori_mesh.compute_vertex_normals()
            partial_pcd = o3d.geometry.PointCloud()
            partial_pcd.points = o3d.utility.Vector3dVector(pcd)
            o3d.visualization.draw_geometries([ori_mesh, partial_pcd])
        pass



if __name__ == "__main__":
    # verts, faces_idx, _ = load_obj("/home/red0orange/Projects/one_shot_il_ws/GraspDiff/tmp/data/cow_mesh/cow.obj")
    # faces = faces_idx.verts_idx
    # verts = np.array(verts)
    # faces = np.array(faces)

    data = np.load("/home/red0orange/Projects/one_shot_il_ws/GraspDiff/mesh.npy", allow_pickle=True).item()
    verts = data["vertices"]
    faces = data["faces"]

    render = Open3dPartialPCDRenderer()
    rot_R = euler_angles_to_matrix(torch.tensor([np.pi / 2, 0, 0]), "XYZ").numpy()
    # rot_R = np.eye(3)
    render.render_partial_pcd(verts, faces, rot_R, debug=True)


    # render = Pytorch3dPartialPCDRenderer(render_distance=0.3)
    # rot_R = euler_angles_to_matrix(torch.tensor([np.pi / 2, 0, 0]), "XYZ").numpy()
    # # rot_R = np.eye(3)
    # render.render_partial_pcd(verts, faces, rot_R, debug=True)

    import time
    for i in range(10):
        time0 = time.time()
        rot_R = euler_angles_to_matrix(torch.tensor([0.0, 0.0, np.pi / 10 * i]), "XYZ").numpy()
        render.render_partial_pcd(verts, faces, rot_R, debug=False)
        print(time.time() - time0)