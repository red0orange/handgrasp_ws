import numpy as np

def calculate_camera_intrinsics(yfov, aspect_ratio, image_width, image_height):
    """
    Calculate camera intrinsic matrix from field of view and aspect ratio.
    
    Parameters:
    - yfov: vertical field of view in radians
    - aspect_ratio: aspect ratio (width/height)
    - image_width: width of the image in pixels
    - image_height: height of the image in pixels
    
    Returns:
    - intrinsics_matrix: 3x3 camera intrinsic matrix
    """
    # Calculate focal lengths using field of view and aspect ratio
    f_y = image_height / (2 * np.tan(yfov / 2))
    f_x = f_y * aspect_ratio
    
    # Assume principal point is at the center of the image
    c_x = image_width / 2
    c_y = image_height / 2
    
    # Construct the intrinsic matrix
    intrinsics_matrix = np.array([[f_x, 0, c_x],
                                  [0, f_y, c_y],
                                  [0, 0, 1]])
    
    return intrinsics_matrix

# Example usage
yfov = 1.042  # 90 degrees field of view
aspect_ratio = 1.0     # Aspect ratio
image_width = 400      # Image width in pixels
image_height = 400     # Image height in pixels

# Calculate the intrinsic matrix
intrinsics_matrix = calculate_camera_intrinsics(yfov, aspect_ratio, image_width, image_height)
print(intrinsics_matrix)




import time
from pytorch3d.renderer import look_at_view_transform
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


mesh = o3d.t.io.read_triangle_mesh("/home/red0orange/Projects/one_shot_il_ws/GraspDiff/tmp/data/cow_mesh/cow.obj")

time0 = time.time()
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(mesh)


height = 512
width = 512
fx = 1000
fy = 1000
cx = width / 2
cy = height / 2

distance = 3  # distance from camera to the object
elevation = 40.0  # angle of elevation in degrees
azimuth = 40.0  # No rotation so the camera is positioned on the +Z axis.

# Get the position of the camera based on the spherical angles
R, T = look_at_view_transform(distance, elevation, azimuth)

R = R[0].numpy()
T = T[0].numpy()

intrinsics = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

pose = np.vstack((np.hstack((R, T[:, None])), np.array([0, 0, 0, 1])))

pose[:2] = pose[:2] * -1

rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
    intrinsic_matrix=intrinsics,
    extrinsic_matrix=pose,
    width_px=int(width),
    height_px=int(height),
)

ans = scene.cast_rays(rays)
print('Time in Open3D {}s'.format(time.time() - time0))

plt.figure()
plt.imshow(ans["t_hit"].numpy())
plt.title("open3d")
plt.show()