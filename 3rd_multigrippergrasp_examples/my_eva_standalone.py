import os
import argparse

def make_parser():
    """ Input Parser """
    parser = argparse.ArgumentParser(description='Standalone script for grasp filtering.')
    parser.add_argument('--exp_name', type=str, help='Directory of Grasp Information', default='')
    return parser

#Parser
parser = make_parser()
input_args = parser.parse_args()


class args:
    force_reset = False
    cache_dir = '/home/red0orange/Projects/handgrasp_ws/0_Data/IsaacSimGraspEvaCache'
    objects_dir = '/home/red0orange/Projects/handgrasp_ws/0_Data/IsaacSimGraspEvaCache/converted_usd'
    gripper_dir = '/home/red0orange/Projects/handgrasp_ws/0_Data/MultiGripperGrasp/grippers'
    # json_dir = '/home/red0orange/projects/docker_isaac_sim_grasping/data/MultiGripperGrasp/graspit_grasps/h5_hand'
    # output_dir = '/home/red0orange/projects/docker_isaac_sim_grasping/data/docker_data/multigrippergrasp_output'

    num_w = 200
    device = 0
    test_time = 3
    print_results = False
    controller = 'position'
    log_level = 'error'
    log_fileLogLevel = 'error'
    log_outputStreamLevel = 'error'

force_reset = args.force_reset
args.exp_name = input_args.exp_name
args.json_dir = os.path.join(args.cache_dir, args.exp_name, "grasps_jsons")
args.output_dir = os.path.join(args.cache_dir, args.exp_name, "output")
os.makedirs(args.output_dir, exist_ok=True)

#launch Isaac Sim before any other imports
from omni.isaac.kit import SimulationApp
config= {
    "headless": False,
    'max_bounces':0,
    'fast_shutdown': True,
    'max_specular_transmission_bounces':0,
    # 'physics_gpu': args.device,
    # 'active_gpu': args.device,

    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,

    "hide_ui": False,
    "renderer": "RayTracedLighting",
    "display_options": 3286,
}
kit = SimulationApp(launch_config=config) # we can also run as headless.

# from omni.isaac.core.utils.extensions import enable_extension
# kit.set_setting("/app/window/drawMouse", True)
# kit.set_setting("/app/livestream/proto", "ws")
# kit.set_setting("/ngx/enabled", False)
# enable_extension("omni.kit.streamsdk.plugins-3.2.1")
# enable_extension("omni.kit.livestream.core-3.2.0")
# enable_extension("omni.kit.livestream.native")

#World Imports
from omni.isaac.core import World
from omni.isaac.core.utils.prims import define_prim
import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.cloner import GridCloner    # import Cloner interface
from omni.isaac.core.utils.stage import add_reference_to_stage

#Omni Libraries
from omni.isaac.core.utils.stage import add_reference_to_stage,open_stage, save_stage
from omni.isaac.core.prims.rigid_prim import RigidPrim 
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.prims import get_prim_children, get_prim_path, get_prim_at_path
from omni.isaac.core.utils.transformations import pose_from_tf_matrix

import omni
import omni.usd
stage = omni.usd.get_context().get_stage()

#External Libraries
import numpy as np
from tqdm import tqdm
import os
import sys
import time
# Custom Classes
# from managers import Manager
# from views import View
from my_managers import Manager
from my_views import View

def import_gripper(work_path,usd_path, EF_axis):
    """ Imports Gripper to World

    Args:
        work_path: prim_path of workstation
        usd_path: path to .usd file of gripper
        EF_axis: End effector axis needed for proper positioning of gripper
    
    """
    T_EF = np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]])
    if (EF_axis == 1):
        T_EF = np.array([[ 0,0,1,0],
                        [ 0,1,0,0],
                        [-1,0,0,0],
                        [0,0,0,1]])
    elif (EF_axis == 2):
        T_EF = np.array([[1, 0,0,0],
                            [0, 0,1,0],
                            [0,-1,0,0],
                            [0, 0,0,1]])
    elif (EF_axis == 3):
        T_EF = np.array([[1, 0, 0,0],
                            [0,-1, 0,0],
                            [0, 0,-1,0],
                            [0, 0, 0,1]])
    elif (EF_axis == -1):
        T_EF = np.array([[0,0,-1,0],
                            [0,1, 0,0],
                            [1,0, 0,0],
                            [0,0, 0,1]])
    elif (EF_axis == -2):
        T_EF = np.array([[1,0, 0,0],
                            [0,0,-1,0],
                            [0,1, 0,0],
                            [0,0, 0,1]])
    #Robot Pose
    gripper_pose= pose_from_tf_matrix(T_EF.astype(float))
    
    # Adding Robot usd
    add_reference_to_stage(usd_path=usd_path, prim_path=work_path+"/gripper")
    robot = world.scene.add(Articulation(prim_path = work_path+"/gripper", name="gripper",
                        position = gripper_pose[0], orientation = gripper_pose[1]))
    robot.set_enabled_self_collisions(False)
    return robot, T_EF

def import_object(work_path, usd_path):
    """ Import Object .usd to World

    Args:
        work_path: prim_path to workstation
        usd_path: path to .usd file of object
    """
    add_reference_to_stage(usd_path=usd_path, prim_path=work_path+"/object")
    object_parent = world.scene.add(GeometryPrim(prim_path = work_path+"/object", name="object"))
    l = get_prim_children(object_parent.prim)
    #print(l)

    # prim = get_prim_at_path(work_path+"/object"+ '/base_link/collisions/mesh_0')
    '''
    MassAPI = UsdPhysics.MassAPI.Get(world.stage, prim.GetPath())
    try: 
        og_mass = MassAPI.GetMassAttr().Get()
        if og_mass ==0:
            og_mass = 1
            print("Failure reading object mass, setting to default value of 1 kg.")
    except:
        og_mass = 1
        print("Failure reading object mass, setting to default value of 1 kg.")

    # Create Rigid Body attribute
    og_mass = 1
    '''

    # all_prims = prims_utils.find_matching_prim_paths(get_prim_path(l[0]) + "*")
    # for prim in all_prims:
    #     print(prim)
    # print(get_prim_path(l[0]))

    prim_path = get_prim_path(l[0])
    prim = stage.GetPrimAtPath(prim_path)
    omni.kit.commands.execute("AddPhysicsComponentCommand",
                                usd_prim=prim,
                                component="PhysicsRigidBodyAPI")

    print("Object prim path: ", prim_path)
    object_prim = GeometryPrim(prim_path=prim_path)
    object_prim.set_collision_enabled(enabled=True)
    object_prim.set_collision_approximation(approximation_type="convexDecomposition")
    object_prim = RigidPrim(prim_path=prim_path)
    # object_prim.set_mass(1)

    mass= 1 #Deprecated use of mass for gravity 

    return object_parent, mass


# @note main
if __name__ == "__main__":
    
    # Directories
    json_directory = args.json_dir
    grippers_directory = args.gripper_dir
    objects_directory = args.objects_dir
    output_directory = args.output_dir
    
    if not os.path.exists(json_directory):
        raise ValueError("Json directory not given correctly")
    elif not os.path.exists(grippers_directory):
        raise ValueError("Grippers directory not given correctly")
    elif not os.path.exists(objects_directory):
        raise ValueError("Objects directory not given correctly")
    elif not os.path.exists(output_directory): 
        raise ValueError("Output directory not given correctly")

    # Testing Hyperparameters
    num_w = args.num_w
    test_time = args.test_time
    verbose = args.print_results
    controller = args.controller
    #physics_dt = 1/120

    world = World(set_defaults = False)
    
    #Debugging
    render = True

    #Load json files 
    json_files = [pos_json for pos_json in os.listdir(json_directory) if pos_json.endswith('.json')]

    for j in json_files:
        print("Running: ", j)

        #path to output .json file
        out_path = os.path.join(output_directory, j)

        if(os.path.exists(out_path)): #Skip completed
            continue

        # Initialize Manager
        # manager 管理要执行的任务相关信息，不涉及 Isaac 仿真部分
        manager = Manager(os.path.join(json_directory,j), grippers_directory, objects_directory, controller)   
        

        #Create initial Workstation Prim
        work_path = "/World/Workstation_0"
        work_prim = define_prim(work_path)

        #Contact names for collisions
        contact_names = []
        for i in manager.c_names:
            contact_names.append(work_path[:-1]+"*"+"/gripper/" +  i)

        #Initialize Workstation
        # 在当前第一个工作空间上添加 gripper 和 object
        robot, T_EF = import_gripper(work_path, manager.gripper_path, manager.EF_axis)
        object_parent, mass = import_object(work_path, manager.object_path)
        
        #Clone
        cloner = GridCloner(spacing = 1)
        target_paths = []
        for i in range(num_w):
            target_paths.append(work_path[:-1]+str(i))
        cloner.clone(source_prim_path = "/World/Workstation_0", prim_paths = target_paths,
                     copy_from_source = True, replicate_physics = True, base_env_path = "/World",
                     root_path = "/World/Workstation_")

        # ISAAC SIM views initialization
        viewer = View(work_path, contact_names, num_w, manager, world, test_time, mass)

        # while True:
        #     world.step(render=render)

        print(robot.dof_names)   # None
        #Reset World and create set first robot positions
        world.reset()
        print(robot.dof_names)   # 有 dof_names，reset 会初始化 Articulation，相当于 art.initialize()

        viewer.dofs, viewer.current_poses, viewer.current_job_IDs = viewer.get_jobs(num_w)

        # Set desired physics Context options
        world.reset()  # 这个 reset 好像没必要？
        physicsContext = world.get_physics_context()
        #physicsContext.set_solver_type("PGS")
        physicsContext.set_physics_dt(manager.physics_dt)
        physicsContext.enable_gpu_dynamics(True)
        physicsContext.enable_stablization(True)
        physicsContext.set_gravity(-9.81)

        world.reset()  # 应该是因为上面的设置会调用一次 step，所以需要再次 reset
        
        # @note 核心的初始化
        viewer.grippers.initialize(world.physics_sim_view)
        viewer.objects.initialize(world.physics_sim_view)
        viewer.post_reset()

        #world.pause()
        #Run Sim
        with tqdm(total=len(manager.completed)) as pbar:
            while not all(manager.completed):
                #print(mass)
                
                world.step(render=render) # execute one physics step and one rendering step if not headless
                # @note 核心的 physics_callback，每次 step 会调用 viewer 的 physics_callback 一次进行核心逻辑的仿真
                #world.pause()

                # tqdm
                if pbar.n != np.sum(manager.completed): #Progress bar
                    pbar.update(np.sum(manager.completed)-pbar.n)
    

        #Save new json with results
        manager.save_json(out_path)
        if (verbose):
            manager.report_results()
        #print("Reseting Environment")

        #Reset World    
        if not force_reset:
            print('Reseting Environment')
            t = time.time()
            world.stop()
            world.clear_physics_callbacks()
            world.clear()
            t = time.time() -t
            print('Reseted, time in seconds: ', t)

        if force_reset:
            os.execl(sys.executable, sys.executable, *sys.argv)
            pass
    
    kit.close() # close Isaac Sim
        
