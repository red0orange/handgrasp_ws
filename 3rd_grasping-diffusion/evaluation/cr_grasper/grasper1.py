"""

Carlyn C. Dougherty
ccd2134@columbia.edu
Robotics Research, Fall 2018

"""

import json
import h5py
import os
import numpy as np
import sys
import pybullet as p
from math import pi, sqrt
import pybullet_data
from time import sleep, time
import astropy.coordinates
import random
from transforms3d import euler
from configparser import ConfigParser
from pyquaternion import Quaternion
import numpy as np
from scipy.spatial import ConvexHull, distance
import trimesh
from scipy.spatial.transform import Rotation
import trimesh
import pybullet_data as pd
import os

# mesh = trimesh.load_mesh(r'G:\Course_Document\MuR_RCI\23ws\Github\pybullet-grasp-master\ObjectURDFs\Mug\mug1.obj')
# mesh.apply_scale(0.0002682458)
# bound = mesh.bounds
# mesh = mesh.convex_hull
# #decompose
# mesh.show()
# p.connect(p.DIRECT)
# name_in = os.path.join(pd.getDataPath(), "duck.obj")
# mesh_duck = trimesh.load_mesh(name_in)
# mesh_duck.show()
# print(f'before {mesh_duck.bounds}')
# name_out = os.path.join(pd.getDataPath(), "duck2.obj")
# name_log = os.path.join(pd.getDataPath(), "duck2.txt")
# p.vhacd(name_in, name_out, name_log, alpha=0.001, resolution=10000000)
# mesh_duck = trimesh.load_mesh(name_out)
# print(f'after {mesh_duck.bounds}')
# mesh_duck.show()

# p.connect(p.DIRECT)
name_in = r'G:\Course_Document\MuR_RCI\23ws\Github\pybullet-grasp-master\ObjectURDFs\Mug\mug1.obj'
name_out = r'G:\Course_Document\MuR_RCI\23ws\Github\pybullet-grasp-master\ObjectURDFs\Mug\mug2.obj'
name_log = "log.txt"
# p.vhacd(name_in, name_out, name_log)
# Assuming you are running in a PyBullet environment already
input_model = name_in
output_model = name_out
mesh_mug = trimesh.load_mesh(name_in)
mesh_mug.apply_scale(0.0002682458)
mesh_mug.bounds
# mesh_mug.show()


# # VHACD parameters can be adjusted based on your needs
# p.vhacd(input_model, output_model,
#         r'G:\Course_Document\MuR_RCI\23ws\Github\pybullet-grasp-master\ObjectURDFs\Mug\mug2.txt')

mesh2 = trimesh.load_mesh(r'G:\Course_Document\MuR_RCI\23ws\Github\pybullet-grasp-master\ObjectURDFs\Mug\mug2.obj')
mesh2.apply_scale(0.0002682458)
bound = mesh2.bounds
# mesh2.show()






PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


#PYBULLET HOUSEKEEPING + GUI MAINTENANCE
physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

# This is to change the visualizer window settings
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, enable=0)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, enable=0)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, enable=0)
# change init camera distance/location to view scene
p.resetDebugVisualizerCamera(cameraDistance=.5, cameraYaw=135, cameraPitch=-20, cameraTargetPosition=[0.0, 0.0, 0.0])



#GLOBAL VARIABLES  - from config file
config = ConfigParser()
print(os.path.join(os.path.dirname(__file__), 'bh_config.ini'))
config.read(os.path.join(os.path.dirname(__file__), 'bh_config.ini'))

robot_path = config.get('file_paths', 'robot_path')
robot_path = os.path.join(PROJECT_ROOT, robot_path)
object_path = config.get('file_paths', 'object_path')
object_path = os.path.join(PROJECT_ROOT, object_path)
object_scale = config.getfloat('file_paths', 'object_scale')
object_file_path = os.path.join(PROJECT_ROOT, "ObjectURDFs/mug/mug.urdf")
# object_file_path = r'G:\Course_Document\MuR_RCI\23ws\Github\pybullet-grasp-master\ObjectURDFs\duck\duck_vhacd.urdf'
grasp_file_path = os.path.join(PROJECT_ROOT, "data/examples/grasps/Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682457830986903.h5")
gripper_path = os.path.join(PROJECT_ROOT, "ObjectURDFs/franka_panda/panda1.urdf")

init_grasp_distance = config.getfloat('grasp_settings', 'init_grasp_distance')
speed_find_distance = config.getfloat('grasp_settings', 'speed_find_distance')
grasp_distance_margin = config.getfloat('grasp_settings', 'grasp_distance_margin')

max_grasp_force = config.getfloat('grasp_settings', 'max_grasp_force')
target_grasp_velocity = config.getfloat('grasp_settings', 'target_grasp_velocity')
grasp_time_limit = config.getfloat('grasp_settings', 'grasp_time_limit')
active_grasp_joints = [int(j.strip()) for j in config.get('grasp_settings', 'active_grasp_joints').split(',')]
num_grasps_per_cycle = config.getint('grasp_settings', 'num_grasps_per_cycle')
num_cycles_to_grasp = config.getint('grasp_settings', 'num_cycles_to_grasp')
num_wrist_rotations = config.getint('grasp_settings', 'num_wrist_rotations')
use_wrist_rotations = config.getboolean('grasp_settings', 'use_wrist_rotations')

force_pyramid_sides = config.getint('eval_settings', 'force_pyramid_sides')
force_pyramid_radius = config.getfloat('eval_settings', 'force_pyramid_radius')

use_gui = config.getboolean('gui_settings', 'use_gui')
debug_lines = config.getboolean('gui_settings', 'debug_lines')
debug_text = config.getboolean('gui_settings', 'debug_text')

"""#####################################################################################################################
                                        Test
#####################################################################################################################"""
panda_robot_path = os.path.join(PROJECT_ROOT, "bullet3/examples/pybullet/gym/pybullet_data/franka_panda/meshes/collision")
finger_obj = os.path.join(panda_robot_path, "finger.obj")
hand_obj = os.path.join(panda_robot_path, "hand.obj")
finger_mesh = trimesh.load(finger_obj)
hand_mesh = trimesh.load(hand_obj)
# combination = trimesh.util.concatenate([finger_mesh, hand_mesh])
# trimesh.Scene([hand_mesh]).show()


"""#####################################################################################################################
                                        Test
#####################################################################################################################"""


# UTILIES
def load_grasps(filename):
    """Load transformations and qualities of grasps from a JSON file from the dataset.

    Args:
        filename (str): HDF5 or JSON file name.

    Returns:
        np.ndarray: Homogenous matrices describing the grasp poses. 2000 x 4 x 4.
        np.ndarray: List of binary values indicating grasp success in simulation.
    """
    if filename.endswith(".json"):
        data = json.load(open(filename, "r"))
        T = np.array(data["transforms"])
        success = np.array(data["quality_flex_object_in_gripper"])
    elif filename.endswith(".h5"):
        data = h5py.File(filename, "r")
        T = np.array(data["grasps/transforms"])
        success = np.array(data["grasps/qualities/flex/object_in_gripper"])
    else:
        raise RuntimeError("Unknown file ending:", filename)
    return T, success
def rand_coord():
    rand_theta = random.uniform(-pi / 2, pi / 2)
    rand_phi = random.uniform(-pi / 2, pi / 2)
    return rand_theta, rand_phi


def add_debug_lines(rID, line_dist=.3, line_width = 500):
    """
    Use pybullet's built-in line functionality to see the z/y/z coords of the hand
    """
    p.addUserDebugLine([0, 0, 0], [line_dist, 0, 0], [1, 0, 0], parentObjectUniqueId=rID, parentLinkIndex=-1,
                       lineWidth=line_width)
    p.addUserDebugLine([0, 0, 0], [0, line_dist, 0], [0, 1, 0], parentObjectUniqueId=rID, parentLinkIndex=-1,
                       lineWidth=line_width)
    p.addUserDebugLine([0, 0, 0], [0, 0, line_dist], [0, 0, 1], parentObjectUniqueId=rID, parentLinkIndex=-1,
                       lineWidth=line_width)


def reset_hand(rID=None, rPos=(0, 0, -init_grasp_distance), rOr=(0, 0, 0, 1), fixed=True):
    """
    move the hand back to the starting pos
    """

    if rID is not None:
        p.removeBody(rID)

    rID = p.loadURDF(gripper_path, basePosition=rPos, baseOrientation=rOr, useFixedBase=fixed, globalScaling=1)
    if debug_lines:
        add_debug_lines(rID)

    return rID


def reset_ob(oID=None, oPos=(0, 0, 0), fixed=True):
    """
    reset by deleting
    """
    if oID is not None:
        p.removeBody(oID)

    oID = p.loadURDF(object_file_path, oPos, globalScaling=1, useFixedBase=fixed)

    return oID

def clean_up(rID):
    """
    remove robot + all associated debug feedback
    """
    p.removeBody(rID)
    p.removeAllUserDebugItems()


"""#####################################################################################################################
                                            HAND ORIENTATION + LOCATION
#####################################################################################################################"""


#TODO: make these more user specifiable (range of angles w reasonable defaults)

def hand_dist(oID, rID, pos, oren):
    """
    actually does tthe movement to have hand touch object to judge distance

    returns position of the hand when it touches the object
    """
    # print("reset hand for non-fixed base")
    reset_hand(rID, rPos=pos, rOr=oren, fixed=False)
    relax(rID)  # want fingers splayed to get distance
    force_vector = np.array(pos)*-speed_find_distance

    has_contact = 0
    while not has_contact:  # while still distance between hand/object
        p.applyExternalForce(rID, -1, force_vector, pos, p.WORLD_FRAME)
        p.stepSimulation()
        has_contact = len(p.getContactPoints(rID, oID))
    t_pos, t_oren = p.getBasePositionAndOrientation(rID)
    clean_up(rID)
    return t_pos  # only need the position of the object


def adjust_point_dist(theta_rad, phi_rad, rID, oID, carts, quat):
    """
    move the hand w/fingers splayed until it touches the object
    should touch in center/palm - this should be the best for an initial grasp

    returns set of position coordinates representing how far from the object the hand should be (touching + a margin)
    """

    t_pos = hand_dist(oID, rID, carts, quat)
    t_dist = distance.euclidean(t_pos, [0,0,0])
    m_dist = t_dist + grasp_distance_margin
    carts = astropy.coordinates.spherical_to_cartesian(m_dist, theta_rad, phi_rad)
    flip_carts = np.array(carts)*-1 #adjust to face obj

    return flip_carts


def get_given_point(dist, theta_rad, phi_rad, rID, oID):
    """
    For the transform3d euler to quat: (seems like their z is our x, their y is our y, their x is our z)
    Rotate about the current z-axis by ϕ. Then, rotate about the new y-axis by θ
    """
    carts = astropy.coordinates.spherical_to_cartesian(dist, theta_rad, phi_rad)
    flip_carts = np.array(carts)*-1 #adjust to face obj
    quat = euler.euler2quat(phi_rad + pi, pi / 2 - theta_rad, pi, axes='sxyz') #pi in the z to face "up"
    close_carts = adjust_point_dist(theta_rad, phi_rad, rID, oID, flip_carts, quat) #find dist to grasp

    return (close_carts, quat)



def sphere_set(rID, oID, phi_init = pi, phi_span = 2*pi, theta_init = pi/2, theta_span = pi/2):
    """
    move the hand around the object in a reasonable way
    returns an array of (position, orientation) pairs
    """
    set = []

    phi = phi_init
    increment_phi = phi_span/num_grasps_per_cycle

    theta = theta_init
    increment_theta = theta_span/num_cycles_to_grasp

    for theta_i in range(0, (num_cycles_to_grasp+1)):
        for phi_i in range(0, (num_grasps_per_cycle)):
            point = get_given_point(dist=init_grasp_distance,
                                    theta_rad=(-theta) + increment_theta * theta_i,
                                    phi_rad=(-phi) + increment_phi * phi_i,
                                    rID=rID, oID=oID)
            set.append(point)

    return set


def rand_set(rID, oID, dist=init_grasp_distance, n=10):
    """
    get n pairs for the hand randomly distributed dist away from the origin
    returns an array of (position, orientation) pairs
    TODO: make this more uniformly distributed https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    """
    set = []

    for i in range(n):
        theta_rad, phi_rad = rand_coord()
        set.append(get_given_point(dist, theta_rad, phi_rad, rID, oID))

    return set


def wrist_rotations(pose):
    """
    rotates the wrist of hand in place, increasing number of grasp possibilities for one position
    """


    rotated_poses = []
    point = pose[0]
    quat = pose[1]
    # change to from xyzw to wxyz
    quat_w = quat[3]
    quat_x = quat[0]
    quat_y = quat[1]
    quat_z = quat[2]
    current_quat = Quaternion(quat_w, quat_x, quat_y, quat_z)
    rot_iter = (2*pi)/num_wrist_rotations
    for i in range(0, num_wrist_rotations):
        rot_quat = Quaternion(axis=np.array(point)*-1, radians=(pi/2)+(rot_iter * i))
        delta_quat = rot_quat * current_quat
        pyb_quat = (delta_quat[1], delta_quat[2], delta_quat[3], delta_quat[0])
        rotated_poses.append((point, pyb_quat))
    return rotated_poses



"""#####################################################################################################################
                                            GRIPPER FUNCTIONS/MOVEMENT
#####################################################################################################################"""


def grasp(handId, max_grasp_force=max_grasp_force, target_grasp_velocity=target_grasp_velocity):
    """
    closes the gripper uniformly + attempts to find a grasp
    this is based on time + not contact points because contact points could just be a finger poking the object
    relies on grip_joints - specified by user/config file which joints should close
    """
    finish_time = time() + 3 #grasp_time_limit
    # input()

    while time() < finish_time:
        p.stepSimulation()
        for joint in [0, 1]:
            p.setJointMotorControl2(bodyUniqueId=handId, jointIndex=joint, controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=target_grasp_velocity,
                                    force=max_grasp_force)
        sleep(1 / 240)


def relax(rID):
    """
    return all joints to neutral/furthest extended, based on urdf specification
    """
    joint = 0
    num = p.getNumJoints(rID)
    while joint < num:
        p.resetJointState(rID, jointIndex=joint, targetValue=0.0)
        joint = joint + 1


"""#####################################################################################################################
                                        POSITION/ORIENTATION DATA - GRAP MEMORY 
#####################################################################################################################"""


class Grasp:

    def __init__(self, robot_position, robot_orientation, robot_joints,
                 final_object_position, final_object_orientation, vol, ep):
        self.robot_pose = (robot_position, robot_orientation)
        self.robot_joints = robot_joints
        self.final_object_pose = (final_object_position, final_object_orientation)
        self.vol = vol
        self.ep = ep

    def __repr__(self):
        return "ROBOT: Pose: " + str(self.robot_pose) + " , Joints: " + str(self.robot_joints) + \
               " OBJECT: Pose: " + str(self.final_object_pose) + \
               " QUALITY: Volume" + str(self.vol)+ " , Epsilon: " + str(self.vol) + " "

    def __str__(self):
        return "ROBOT: Pose: " + str(self.robot_pose) + " , Joints: " + str(self.robot_joints) + \
               " OBJECT: Pose: " + str(self.final_object_pose) + \
               " QUALITY: Volume" + str(self.vol)+ " , Epsilon: " + str(self.vol) + " "

def get_robot_config(rID, oID):
    r_pos, r_oren = p.getBasePositionAndOrientation(rID)
    joints = {}
    num = p.getNumJoints(rID)
    for joint in range(0, num):
        joints[joint] = p.getJointState(rID, joint)
    o_pos, o_oren = p.getBasePositionAndOrientation(oID)
    vol, ep = grip_qual(rID,oID)
    return Grasp(r_pos, r_oren, joints, o_pos, o_oren, vol, ep)


"""#####################################################################################################################
                                            GRASP EVALUATION FUNCTIONS
#####################################################################################################################"""


def check_grip(oID, rID):
    """
    check grip by adding in gravity
    """
    #print("checking strength of current grip")
    mag = 1
    pos, oren = p.getBasePositionAndOrientation(rID)
    time_limit = .5
    finish_time = time() + time_limit
    p.addUserDebugText("Grav Check!", [-.07, .07, .07], textColorRGB=[0, 0, 1], textSize=1)
    while time() < finish_time:
        p.stepSimulation()
        p.applyExternalForce(oID, linkIndex=-1, forceObj=[0, 0, -mag], posObj=pos, flags=p.WORLD_FRAME)
    contact = p.getContactPoints(oID, rID)  # see if hand is still holding obj after gravity is applied
    if len(contact) > 0:
        p.removeAllUserDebugItems()
        p.addUserDebugText("Grav Check Passed!", [-.07, .07, .07], textColorRGB=[0, 1, 0], textSize=1)
        print("Grav Check Passed")
        sleep(.2)
        return get_robot_config(rID, oID)
    else:
        p.removeAllUserDebugItems()
        p.addUserDebugText("Grav Check Failed!", [-.07, .07, .07], textColorRGB=[1, 0, 0], textSize=1)
        print("Grav Check Failed")
        sleep(.2)
        return None


def grip_qual(oID, rID):
    """
    evaluate the grasp quality
    """
    contact = p.getContactPoints(oID, rID)  # see if hand is still holding obj after gravity is applied
    if len(contact) > 0:
        force_torque = gws_pyramid_extension(rID, oID)
        #print("force_torque: ", force_torque)
        #print("force_torque shape: ", np.array(force_torque).shape)
        vol = volume(force_torque)
        #print("volume: ", vol)
        ep = eplison(force_torque)
        #print("eplison: ", ep)
    else:
        vol = None
        ep = None
    return vol, ep


def get_obj_info(oID): #TODO: what about not mesh objects?
    """
    get object data to figure out how far away the hand needs to be to make its approach
    """
    obj_data = p.getCollisionShapeData(oID, -1)[0]
    geometry_type = obj_data[2]
    #print("geometry type: " + str(geometry_type))
    dimensions = obj_data[3]
    #print("dimensions: "+ str(dimensions))
    local_frame_pos = obj_data[5]
    #print("local frome position: " + str(local_frame_pos))
    local_frame_orn = obj_data[6]
    #print("local frame oren: " + str(local_frame_orn))
    diagonal = sqrt(dimensions[0]**2+dimensions[1]**2+dimensions[2]**2)
    #print("diagonal: ", diagonal)
    max_radius = diagonal/2
    return local_frame_pos, max_radius


def gws(rID, oID):
    """
    calculate force/torque vectors for use in evaluation

    """
    print("eval gws")
    local_frame_pos, max_radius = get_obj_info(oID)
    #sim uses center of mass as a reference for the Cartesian world transforms in getBasePositionAndOrientation
    obj_pos, obj_orn = p.getBasePositionAndOrientation(oID)
    force_torque = []
    contact_points = p.getContactPoints(rID, oID)
    for point in contact_points:
        contact_pos = point[6]
        normal_vector_on_obj = point[7]
        normal_force_on_obj = point[9]
        force_vector = np.array(normal_vector_on_obj)*normal_force_on_obj


        radius_to_contact = np.array(contact_pos) - np.array(obj_pos)
        torque_numerator = np.cross(radius_to_contact, force_vector)
        torque_vector = torque_numerator/max_radius

        force_torque.append(np.concatenate([force_vector, torque_vector]))

    return force_torque


def get_new_normals(force_vector, normal_force, sides, radius):
    """
    utility function to help with GWS/pyramid extension for contact points
    """
    return_vectors = []
    #get arbitrary vector to get cross product which should be orthogonal to both
    vector_to_cross = np.array((force_vector[0]+1,force_vector[1]+2,force_vector[2]+3))
    orthg = np.cross(force_vector, vector_to_cross)
    orthg_vector = (orthg/np.linalg.norm(orthg))*radius
    rot_angle = (2*pi)/sides
    split_force = normal_force/sides

    for side_num in range(0,sides):
        rotated_orthg = Quaternion(axis=force_vector, angle=(rot_angle*side_num)).rotate(orthg_vector)
        new_vect = force_vector+np.array(rotated_orthg)
        norm_vect = (new_vect/np.linalg.norm(new_vect))*split_force
        return_vectors.append(norm_vect)

    return return_vectors


def gws_pyramid_extension(rID, oID, pyramid_sides=force_pyramid_sides, pyramid_radius=force_pyramid_radius):
    #often dont have enough contact points to create a qhull of the right dimensions, so create more that are very close to the existing ones
    local_frame_pos, max_radius = get_obj_info(oID)
    #sim uses center of mass as a reference for the Cartesian world transforms in getBasePositionAndOrientation
    obj_pos, obj_orn = p.getBasePositionAndOrientation(oID)
    force_torque = []
    contact_points = p.getContactPoints(rID, oID)
    for point in contact_points:
        contact_pos = point[6]
        normal_vector_on_obj = point[7]
        normal_force_on_obj = point[9]
        force_vector = np.array(normal_vector_on_obj)*normal_force_on_obj
        if np.linalg.norm(force_vector) > 0:
            new_vectors = get_new_normals(force_vector, normal_force_on_obj, pyramid_sides, pyramid_radius)

            radius_to_contact = np.array(contact_pos) - np.array(obj_pos)

            for pyramid_vector in new_vectors:
                torque_numerator = np.cross(radius_to_contact, pyramid_vector)
                torque_vector = torque_numerator/max_radius
                force_torque.append(np.concatenate([pyramid_vector, torque_vector]))

    return force_torque


def volume(force_torque):
    """
    get qhull of the 6 dim vectors [fx, fy, fz, tx, ty, tz] created by gws (from contact points)
    get the volume
    """
    vol = ConvexHull(points = force_torque)
    return vol.volume


def eplison(force_torque):
    """
    get qhull of the 6 dim vectors [fx, fy, fz, tx, ty, tz] created by gws (from contact points)
    get the distance from centroid of the hull to the closest vertex
    """
    hull = ConvexHull(points=force_torque)
    centroid = []
    for dim in range(0,6):
        centroid.append(np.mean(hull.points[hull.vertices, dim]))
    shortest_distance = 500000000
    closest_point = None
    for point in force_torque:
        point_dist = distance.euclidean(centroid, point)
        if point_dist < shortest_distance:
            shortest_distance = point_dist
            closest_point = point

    return shortest_distance

# Function to convert position and orientation (as quaternion) to a 4x4 matrix
def pose_to_mat(position, orientation):
    mat = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    mat = np.pad(mat, ((0, 1), (0, 1)), mode='constant')
    mat[:, 3] = list(position) + [1]
    return mat

# Function to convert a 4x4 matrix to position and orientation (as quaternion)
def mat_to_pose(matrix):
    position = matrix[:3, 3]
    # Create a Rotation object from the matrix
    rot = Rotation.from_matrix(matrix[:3, :3])

    # Convert to quaternion
    orientation = rot.as_quat()  # Returns a quaternion [x, y, z, w]
    # orientation = p.getQuaternionFromEuler(p.getQuaternionFromRotationMatrix(matrix[:3, :3]))
    return position, orientation

def create_gripper_marker(color=[0, 0, 255], tube_radius=0.001, sections=6):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [4.10000000e-02, -7.27595772e-12, 6.59999996e-02],
            [4.10000000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [-4.100000e-02, -7.27595772e-12, 6.59999996e-02],
            [-4.100000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=0.002, sections=sections, segment=[[0, 0, 0], [0, 0, 6.59999996e-02]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[[-4.100000e-02, 0, 6.59999996e-02], [4.100000e-02, 0, 6.59999996e-02]],
    )
    cub = trimesh.creation.icosphere(3, 0.005)
    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl, cub])
    tmp.visual.face_colors = color
    # trimesh.Scene(tmp).show()

    return tmp
"""#####################################################################################################################
                                        MAIN MAIN MAIN MAIN MAIN MAIN
#####################################################################################################################"""

T, success = load_grasps(grasp_file_path)
t_grasps = np.squeeze(T[np.random.choice(np.where(success == 1)[0], 100)])
# t_grasps = np.squeeze(T[np.where(success == 1)[0][:10]])
T_success = T[np.where(success == 1)[0]]
# rID = reset_hand(rPos=(0, 0, 0), rOr=(0.707,0,0,0.707), fixed=True)
rID = reset_hand(rPos=(0, 0, 0), rOr=(0, 0, 0.707, 0.707), fixed=True)
# x_oID = reset_ob(fixed=True)
# p.resetBasePositionAndOrientation(rID, [0, 0, 0], [0.707, 0.707, 0, 0])
def set_ob():

    meshFilePath = r'G:\Course_Document\MuR_RCI\23ws\Github\pybullet-grasp-master\ObjectURDFs\Mug\mug2.obj'
    collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                              fileName=meshFilePath,
                                              meshScale=[0.0002682458, 0.0002682458, 0.0002682458])
    # Create the multi-body object using the collision shape
    objectUniqueId = p.createMultiBody(baseMass=0.0130879738,  # Set mass to 0 for static objects
                                       baseCollisionShapeIndex=collisionShapeId,
                                       basePosition=[0, 0, 0],  # Initial position
                                       baseOrientation=[0, 0, 0, 1])  # Initial orientation as a quaternion
    # Set dynamic parameters for the object
    p.changeDynamics(objectUniqueId, -1,  # -1 applies the changes to the base link
                     lateralFriction=1.0,  # Friction coefficient
                     spinningFriction=1,  # Friction while spinning
                     rollingFriction=1,  # Friction while rolling
                     restitution=0.5,  # Bounciness
                     linearDamping=0.04,  # Linear damping
                     angularDamping=0.04)  # Angular damping
      # <lateral_friction value="1.0"/>
      # <rolling_friction value="1.0"/>
      # <inertia_scaling value="1.0"/>
      # <contact_cfm value="1.0"/>
      # <contact_erp value="1.0"/>

    return objectUniqueId

# mesh = trimesh.load_mesh(r'G:\Course_Document\MuR_RCI\23ws\Github\pybullet-grasp-master\ObjectURDFs\Mug\mug2.obj')
# mesh = mesh.apply_scale(0.0002682458)
# mesh.show()
# mesh2.show()
oID = set_ob()
# Create a fixed constraint that locks the object in place
# The parameters are (parentBodyUniqueId, parentLinkIndex, childBodyUniqueId, childLinkIndex,
# jointType, jointAxis, parentFramePosition, childFramePosition, parentFrameOrientation, childFrameOrientation)
constraint_id = p.createConstraint(parentBodyUniqueId=oID,
                                   parentLinkIndex=-1,  # -1 refers to the base link of the object
                                   childBodyUniqueId=-1,  # No child body, so set to -1
                                   childLinkIndex=-1,  # No child link, so set to -1
                                   jointType=p.JOINT_FIXED,  # Fixed joint type to lock the position and orientation
                                   jointAxis=[0, 0, 0],  # Joint axis is irrelevant for fixed joints
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0],
                                   parentFrameOrientation=[0, 0, 0, 1],
                                   childFrameOrientation=[0, 0, 0, 1])

# Now the object is fixed in place and will not react to physics
p.removeConstraint(constraint_id)
linearDamping = 1
angularDamping = 1
dynamicsInfo1 = p.getDynamicsInfo(oID, -1)
p.changeDynamics(oID, linkIndex=-1, linearDamping=linearDamping, angularDamping=angularDamping)
dynamicsInfo = p.getDynamicsInfo(oID, -1)
gravity_force = [0, 0, -dynamicsInfo[0] * 9.8]
max_grasp_force = 6 * dynamicsInfo[0] * 9.8
target_grasp_velocity = 0.3

for i, t_grasp in enumerate(t_grasps):
    print(f"grasp:{i}!")
    successful_grasp = [
        create_gripper_marker(color=[0, 255, 0]).apply_transform(t_grasp)
    ]
    initial_grasp = [create_gripper_marker(color=[0, 0, 255])]
    # trimesh.Scene([mesh2] + initial_grasp).show()
    # t_grasp = np.squeeze(T[np.random.choice(np.where(success == 1)[0], 1)])
    # rID = reset_hand(rPos=(0, 0, 0), rOr=(0, 0, 0, 1), fixed=True)
    # current_position, current_orientation = p.getBasePositionAndOrientation(rID)
    current_pose_mat = pose_to_mat((0, 0, 0), (0, 0, 0.707, 0.707))
    p.resetBasePositionAndOrientation(rID, (0, 0, 0), (0, 0, 0.707, 0.707))
    # Combine the current pose with the transformation
    new_pose_mat = np.dot(t_grasp, current_pose_mat)
    # Extract the new position and orientation
    new_position, new_orientation = mat_to_pose(new_pose_mat)

    # Apply the new pose and position
    p.removeAllUserDebugItems()
    p.resetBasePositionAndOrientation(rID, new_position, new_orientation)

    # oID = reset_ob(oID, fixed=False)
    # oID = set_ob()
    p.resetBasePositionAndOrientation(oID, [0, 0, 0], [0, 0, 0, 1])
    linearDamping = 0.5
    angularDamping = 0.5
    p.changeDynamics(oID, linkIndex=-1, linearDamping=linearDamping, angularDamping=angularDamping)
    dynamicsInfo1 = p.getDynamicsInfo(oID, -1)
    relax(rID)

    # check if the pre-grasp position is correct

    # trimesh.Scene([mesh2] + successful_grasp).show()
    contact = p.getContactPoints(oID, rID)
    if len(contact) > 0:
        print('wrong!')
    grasp(rID, max_grasp_force=max_grasp_force, target_grasp_velocity=target_grasp_velocity)
    finish_time = time() + 1
    p.addUserDebugText("Grav Check!", [-.07, .07, .07], textColorRGB=[0, 0, 1], textSize=1)
    linearDamping = 0.04
    angularDamping = 0.04
    p.changeDynamics(oID, linkIndex=-1, linearDamping=linearDamping, angularDamping=angularDamping)
    # print(time())
    # print(time())
    # print(time())
    # # finish_time = time() + 3
    # print(finish_time)
    # max_grasp_force = dynamicsInfo[0] * 9.8
    while time() < finish_time:
        begin = time()

        p.stepSimulation()
        for joint in [0, 1]:
            p.setJointMotorControl2(bodyUniqueId=rID, jointIndex=joint, controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=target_grasp_velocity,
                                    force=max_grasp_force)
        info = p.getBasePositionAndOrientation(oID)
        p.applyExternalForce(oID, linkIndex=-1, forceObj=gravity_force, posObj=info[0], flags=p.WORLD_FRAME)
        end = time()
        sleep(1 / 240)
        # print(f'interval {end - begin}')
    contact = p.getContactPoints(oID, rID)
    if len(contact) > 0:
        p.removeAllUserDebugItems()
        p.addUserDebugText("Grav Check Passed!", [-.07, .07, .07], textColorRGB=[0, 1, 0], textSize=1)
        print("Grav Check Passed")
        sleep(1)




# print("Sphere Set")
# hand_set = sphere_set(rID=rID, oID=oID)
# #hand_set = rand_set(rID=handID, oID=cubeID, n=45)
# print(hand_set)
#
# rID = reset_hand()
# oID = reset_ob(oID, [0, 0, 0])
#
# good_grips = []
#
# pos = 0
# for pose in hand_set:
#     poses = []
#     poses.append(pose)
#     if use_wrist_rotations:
#         rotated_poses = wrist_rotations(pose)
#         poses = poses + rotated_poses
#
#     for pose in poses:
#         print(" ")
#         print("Pose #: ", pos)
#         relax(rID)
#         p.removeAllUserDebugItems()
#         p.resetBasePositionAndOrientation(rID, pose[0], pose[1])
#         if debug_lines:
#             add_debug_lines(rID)
#         oID = reset_ob(oID, [0, 0, 0], fixed=False)
#         grasp(rID)
#         vol, ep = grip_qual(oID, rID)
#         print("Volume: ", vol)
#         print("Epslion: ", ep)
#         good_grips.append(check_grip(oID, rID))
#         pos += 1
#
#
# print("Num Good Grips: ", len(good_grips))
# print("Grips:")
# for grip in good_grips:
#     print(grip)

