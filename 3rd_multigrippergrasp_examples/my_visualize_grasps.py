#External Libraries
import numpy as np
from tqdm import tqdm
import os
import argparse
import time

"""
./python.sh (repo directory)/visualize_grasps.py --json_dir=(dataset .json folder) --gripper_dir=(repo directory)/grippers --objects_dir=(object .usd folder) --num_w=10 --ub=3 --lb=0 --/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error
"""

device = 0
controller = "default"

def make_parser():
    """ Input Parser """
    parser = argparse.ArgumentParser(description='Visualization script for filtered grasps.')
    parser.add_argument('--transfer', type=bool, help='Indicate if file is a transfer file',
                         default=False, action = argparse.BooleanOptionalAction)
    parser.add_argument('--/log/level', type=str, help='isaac sim logging arguments', default='', required=False)
    parser.add_argument('--/log/fileLogLevel', type=str, help='isaac sim logging arguments', default='', required=False)
    parser.add_argument('--/log/outputStreamLevel', type=str, help='isaac sim logging arguments', default='', required=False)
    
    return parser

#Parser
parser = make_parser()
args = parser.parse_args()
transfer = args.transfer

#launch Isaac Sim before any other imports
import isaacsim
from omni.isaac.kit import SimulationApp
# from omni.isaac.kit import SimulationApp
config= {
    "headless": False,
    'max_bounces':0,
    'fast_shutdown': True,
    'max_specular_transmission_bounces':0,
    'physics_gpu': device,
    'active_gpu': device
    }
simulation_app = SimulationApp(config) # we can also run as headless.


#World Imports
from omni.isaac.core import World
from omni.isaac.core.utils.prims import define_prim
from omni.isaac.cloner import GridCloner    # import Cloner interface
from omni.isaac.core.utils.stage import add_reference_to_stage

# Custom Classes
from views import V_View
from managers import V_Manager

#Omni Libraries
from omni.isaac.core.utils.stage import add_reference_to_stage,open_stage, save_stage
from omni.isaac.core.prims.rigid_prim import RigidPrim 
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.prims import get_prim_children, get_prim_path, get_prim_at_path
from omni.isaac.core.utils.transformations import pose_from_tf_matrix
import omni.isaac.core.utils.prims as prim_utils



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

    prim = get_prim_at_path(work_path+"/object"+ '/base_link/collisions/mesh_0')
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

    object_prim = RigidPrim(prim_path= get_prim_path(l[0]))

    mass= 1 #Deprecated use of mass for gravity 

    return object_parent, mass


if __name__ == "__main__":
    # Testing Hyperparameters
    num_w = 10
    lb = 0
    ub = 3

    # Directories
    json_directory = "/home/red0orange/Data/MultiGripperGrasp/Dataset-selected/graspit_grasps/HumanHand"
    grippers_directory = "/home/red0orange/Projects/one_shot_il_ws/isaac_sim_grasping/grippers"
    objects_directory = "/home/red0orange/Data/MultiGripperGrasp/Dataset-selected/google_objects_usd"
    
    if not os.path.exists(json_directory):
        raise ValueError("Json directory not given correctly")
    elif not os.path.exists(grippers_directory):
        raise ValueError("Grippers directory not given correctly")
    elif not os.path.exists(objects_directory):
        raise ValueError("Objects directory not given correctly")

    if (transfer and controller == 'default'):
        controller = 'transfer_default'
    else:
        controller = controller

    world = World(set_defaults = False)
    
    #Debugging
    render = True

    #Load json files 
    json_files = [pos_json for pos_json in os.listdir(json_directory) if pos_json.endswith('.json')]
    print(json_files)

    for j in json_files:
        # Initialize Manager
        manager = V_Manager(os.path.join(json_directory,j), grippers_directory, objects_directory, controller, transfer, ub = ub, lb = lb)
        
        #Create initial Workstation Prim
        work_path = "/World/Workstation_0"
        work_prim = define_prim(work_path)

        #Contact names for collisions
        contact_names = []
        for i in manager.c_names:
            contact_names.append(work_path[:-1]+"*"+"/gripper/" +  i)

        #Initialize Workstation
        robot, T_EF = import_gripper(work_path, manager.gripper_path,manager.EF_axis)
        object_parent, mass = import_object(work_path, manager.object_path)
        
        #Clone
        cloner = GridCloner(spacing = 1)
        target_paths = []
        for i in range(num_w):
             target_paths.append(work_path[:-1]+str(i))
        cloner.clone(source_prim_path = "/World/Workstation_0", prim_paths = target_paths,
                     copy_from_source = True, replicate_physics = True, base_env_path = "/World",
                     root_path = "/World/Workstation_")
        
        light_1 = prim_utils.create_prim(
            "/World/Light_1",
            "DomeLight",
            attributes={
                "inputs:intensity": 1000
            }
        )

        # ISAAC SIM views initialization
        viewer = V_View(work_path,contact_names,num_w, manager,world, mass)

        
        #Reset World and create set first robot positions
        world.reset()

        # Print Robot DoFs
        print(robot.dof_names)
        viewer.dofs, viewer.current_poses, viewer.current_job_IDs = viewer.get_jobs(num_w)

        # Set desired physics Context options
        world.reset()
        physicsContext = world.get_physics_context()
        physicsContext.set_physics_dt(manager.physics_dt)
        physicsContext.enable_gpu_dynamics(True)
        physicsContext.enable_stablization(True)
        physicsContext.set_gravity(0)

        world.reset()
        
        #Initialize views
        viewer.grippers.initialize(world.physics_sim_view)
        viewer.objects.initialize(world.physics_sim_view)
        viewer.post_reset()

        #world.pause()
        #Run Sim
        with tqdm(total=len(manager.completed)) as pbar:
            #world.pause()
            while not all(manager.completed):
                world.step(render=render) # execute one physics step and one rendering step if not headless
                if pbar.n != np.sum(manager.completed): #Progress bar
                    pbar.update(np.sum(manager.completed)-pbar.n)

        print('Reseting Environment')
        t = time.time()
        world.stop()
        world.clear_physics_callbacks()
        world.clear()
        t = time.time() -t
        print('Reseted, time in seconds: ', t)
    
    simulation_app.close() # close Isaac Sim
        
