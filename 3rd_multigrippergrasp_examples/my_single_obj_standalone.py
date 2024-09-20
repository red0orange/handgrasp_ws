class args:
    # headless = True
    headless = False
    force_reset = False
    json_dir = '/home/red0orange/Data/MultiGripperGrasp/Dataset-selected/graspit_grasps/HumanHand'
    # json_dir = '/home/red0orange/Data/MultiGripperGrasp/Dataset-selected/graspit_grasps/h5_hand'
    # json_dir = '/home/red0orange/Data/MultiGripperGrasp/Dataset-selected/graspit_grasps/franka_panda'
    gripper_dir = '/home/red0orange/Projects/one_shot_il_ws/isaac_sim_grasping/grippers'
    objects_dir = '/home/red0orange/Data/MultiGripperGrasp/Dataset-selected/google_objects_usd'
    output_dir = '/home/red0orange/Data/MultiGripperGrasp/output'

    num_w = 2
    device = 0
    test_time = 3
    print_results = False
    controller = 'position'
    log_level = 'error'
    log_fileLogLevel = 'error'
    log_outputStreamLevel = 'error'

head = args.headless
force_reset = args.force_reset
print(args.controller)

#launch Isaac Sim before any other imports
import isaacsim
from omni.isaac.kit import SimulationApp
config= {
    "headless": head,
    'max_bounces':0,
    'fast_shutdown': True,
    'max_specular_transmission_bounces':0,
    'physics_gpu': args.device,
    'active_gpu': args.device
    }
simulation_app = SimulationApp(launch_config=config) # we can also run as headless.

#World Imports
from omni.isaac.core import World
from omni.isaac.core.utils.prims import define_prim
from omni.isaac.cloner import GridCloner    # import Cloner interface
from omni.isaac.core.utils.stage import add_reference_to_stage

#Omni Libraries
from omni.isaac.core.utils.stage import add_reference_to_stage,open_stage, save_stage
from omni.isaac.core.prims.rigid_prim import RigidPrim 
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.prims import get_prim_children, get_prim_path, get_prim_at_path
from omni.isaac.core.utils.transformations import pose_from_tf_matrix
from omni.isaac.core.prims import RigidPrimView

#External Libraries
import numpy as np
from tqdm import tqdm
import os
import sys
import time
# Custom Classes
from managers import Manager
from views import View

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


class MyPhysicsCallback:
    def __init__(self, viewer):
        self.viewer = viewer
        self.objects = viewer.objects
        pass

    def post_reset(self):
        pass

    def physics_callback(self, _):
        # print("Debug: My Physics Callback")

        # tangential forces
        friction_forces, friction_points, friction_pair_contacts_count, friction_pair_contacts_start_indices = self.objects.get_friction_data(dt=1 / 60)
        # normal forces
        forces, points, normals, distances, pair_contacts_count, pair_contacts_start_indices = self.objects.get_contact_force_data(dt=1 / 60)
        cur_objs_state = self.objects.get_current_dynamic_state()
        print("points: {}, object_center: {}".format(points[0], cur_objs_state.positions[0]))

        force_aggregate = np.zeros((self.objects._contact_view.num_shapes, self.objects._contact_view.num_filters, 3))
        friction_force_aggregate = np.zeros((self.objects._contact_view.num_shapes, self.objects._contact_view.num_filters, 3))

        # process contacts for each pair i, j
        for i in range(pair_contacts_count.shape[0]):
            for j in range(pair_contacts_count.shape[1]):
                start_idx = pair_contacts_start_indices[i, j]
                friction_start_idx = friction_pair_contacts_start_indices[i, j]
                count = pair_contacts_count[i, j]
                friction_count = friction_pair_contacts_count[i, j]
                # sum/average across all the contact points for each pair
                pair_forces = forces[start_idx : start_idx + count]
                pair_normals = normals[start_idx : start_idx + count]
                force_aggregate[i, j] = np.sum(pair_forces * pair_normals, axis=0)

                # sum/average across all the friction pairs
                pair_forces = friction_forces[friction_start_idx : friction_start_idx + friction_count]
                friction_force_aggregate[i, j] = np.sum(pair_forces, axis=0)

        # print("friction forces: \n", friction_force_aggregate)
        # print("contact forces: \n", force_aggregate)
        # get_contact_force_matrix API is equivalent to the summation of the individual contact forces computed above
        # print("contact force matrix: \n", self.objects.get_contact_force_matrix(dt=1 / 60))
        # get_net_contact_forces API is the summation of the all forces
        # in the current example because all the potential contacts are captured by the choice of our filter prims (/World/defaultGroundPlane/GroundPlane/CollisionPlane)
        # the following is similar to the reduction of the contact force matrix above across the filters
        # print("net contact force: \n", self.objects.get_net_contact_forces(dt=1 / 60))
        pass


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
    render = not head

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

        # 添加我的 physics_callback
        my_physics_callback = MyPhysicsCallback(viewer)
        world.add_physics_callback("my_phy_cb", my_physics_callback.physics_callback)
        
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

        my_physics_callback.post_reset()

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
    
    simulation_app.close() # close Isaac Sim
        
