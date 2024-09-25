import os
import time
import os.path as osp
import math
import copy

from isaac_evaluation.utils.geometry_utils import Transform_2_H, H_2_Transform
from isaacgym import gymapi, gymtorch
from isaacgym import gymutil

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from isaac_evaluation.utils.generate_obj_urdf import generate_obj_urdf
from isaac_evaluation.utils.geometry_utils import pq_to_H

cur_dir = os.path.dirname(os.path.abspath(__file__))
TABLE_PATH = os.path.join(os.path.dirname(cur_dir), "grasp_sim/meshes/table/table.urdf")
FRANKA_URDF_PATH = os.path.join(os.path.dirname(cur_dir), "grasp_sim/meshes/hand_only/robots/franka_panda_hand.urdf")


class AnyGraspSuccessEvaluator():
    """暂定输入：一个物体模型和要评估的抓取列表，返回成功率
    """
    def __init__(self, obj_mesh_path, rotations=None, scales=None, 
                 n_envs = 10, viewer=True, device='cuda:0', enable_rel_trafo=True):
        self.device = device
        self.n_envs = n_envs
        # This argument tells us if the grasp poses are relative w.r.t. the current object pose or not
        self.enable_rel_trafo = enable_rel_trafo

        self.obj_mesh_path = obj_mesh_path
        self.obj_name = os.path.basename(obj_mesh_path).split('.')[0]
        if scales is None:
            scales = [1.]*n_envs
        if rotations is None:
            rotations = [[0,0,0,1]]*n_envs

        ## Build Envs ##
        env_args = self._get_args(self.obj_mesh_path, self.obj_name, scales, rotations)

        self.grasping_env = AnyIsaacGymWrapper(env=AnyGraspingGymEnv, env_args=env_args,
                                            z_convention=True, num_spaces=self.n_envs,
                                            viewer = viewer, device=self.device)

        self.success_cases = 0
        self.batch_idx = 0

    def reset(self):
        self.success_cases = 0

    def _get_args(self, obj_mesh_path, obj_name, scales, rotations):
        args = []
        for i in range(self.n_envs):
            obj_args = {
                "obj_mesh_path": obj_mesh_path,
                "obj_name": obj_name,
                "scale": scales[i],
                "obj_or": rotations[i]
            }
            args.append({"obj_args": obj_args})

        return args

    def eval_set_of_grasps(self, H):
        n_grasps = H.shape[0]
        self.success_flags = np.zeros((n_grasps // self.n_envs) * self.n_envs + self.n_envs)

        self.batch_idx = 0
        for i in range(0, n_grasps, self.n_envs):
            print('iteration: {}'.format(i))

            batch_H = H[i:i+self.n_envs,...]

            if batch_H.shape[0] < self.n_envs:
                fake_H = torch.eye(4, device=self.device)[None,...].repeat(self.n_envs-batch_H.shape[0], 1, 1)
                fake_H[:, :3, 3] = torch.tensor([0,0,0.15], device=self.device)
                batch_H = torch.cat([batch_H, fake_H], dim=0)
            
            self.eval_batch(batch_H)
            self.batch_idx += 1

        return self.success_cases, self.success_flags[:n_grasps]

    def eval_batch(self, H):

        s = self.grasping_env.reset()
        for t in range(10):
            self.grasping_env.step()
        s = self.grasping_env.reset()
        for t in range(10):
            self.grasping_env.step()

        # 1. Set Evaluation Grasp
        H_obj = torch.zeros_like(H)
        for i, s_i in enumerate(s):
            H_obj[i,...] = pq_to_H(p=s_i['obj_pos'], q=s_i['obj_rot'])
            if not(self.enable_rel_trafo):
                H_obj[i, :3, :3] = torch.eye(3)
        Hg = torch.einsum('bmn,bnk->bmk', H_obj, H)

        state_dicts = []
        for i in range(Hg.shape[0]):
            state_dict = {
                'grip_state': Hg[i,...]
            }
            state_dicts.append(state_dict)

        s = self.grasping_env.reset(state_dicts)

        # 2. Grasp
        policy = AnyGraspController(self.grasping_env, n_envs=self.n_envs)

        # # @note 
        # while True:
        #     self.grasping_env.only_render()

        T = 700
        for t in range(T):
            a = policy.control(s)
            s = self.grasping_env.step(a)

        self._compute_success(s)
        del policy
        torch.cuda.empty_cache()

    def _compute_success(self, s):
        for i, si in enumerate(s):
            hand_pos = si['hand_pos']
            obj_pos  = si['obj_pos']
            ## Check How close they are ##
            distance = (hand_pos - obj_pos).pow(2).sum(-1).pow(.5)

            # @note BUG
            # print('Distance: {}'.format(distance))
            # if distance <2.0:
            if distance <0.3:
                self.success_flags[self.batch_idx * self.n_envs + i] = 1
                self.success_cases +=1



class AnyGraspController():
    '''
     A controller to evaluate the grasping
    '''
    def __init__(self, env, hand_cntrl_type='position', finger_cntrl_type='torque', n_envs = 0):
        self.env = env

        ## Controller Type
        self.hand_cntrl_type = hand_cntrl_type
        self.finger_cntrl_type = finger_cntrl_type

        self.squeeze_force = .6
        self.hold_force = [self.squeeze_force]*n_envs

        self.r_finger_target = [0.]*n_envs
        self.l_finger_target = [0.]*n_envs
        self.grasp_count = [0]*n_envs

        ## State Machine States
        self.control_states = ['approach', 'grasp', 'lift']
        self.grasp_states = ['squeeze', 'hold']

        self.state = ['grasp']*n_envs
        self.grasp_state = ['squeeze']*n_envs

    def set_H_target(self, H):
        self.H_target = H
        self.T = H_2_Transform(H)

    def control(self, states):
        actions = []
        for idx, state in enumerate(states):
            if self.state[idx] =='approach':
                action = self._approach(state, idx)
            elif self.state[idx] == 'grasp':
                if self.grasp_state[idx] == 'squeeze':
                    action = self._squeeze(state, idx)
                elif self.grasp_state[idx] == 'hold':
                    action = self._hold(state, idx)
            elif self.state[idx] == 'lift':
                action = self._lift(state, idx)

            actions.append(action)
        return actions

    def _approach(self, state, idx=0):
        hand_pose  = state[0]['hand_pos']
        hand_rot  = state[0]['hand_rot']

        target_pose = torch.Tensor([self.T.p.x, self.T.p.y, self.T.p.z])

        pose_error = hand_pose - target_pose
        des_pose = hand_pose - pose_error*0.1

        ## Desired Pos for left and right finger is 0.04
        r_error = state[0]['r_finger_pos'] - .04
        l_error = state[0]['l_finger_pos'] - .04

        K = 1000
        D = 20
        ## PD Control law for finger torque control
        des_finger_torque = torch.zeros(2)
        des_finger_torque[1:] += -K*r_error - D*state[0]['r_finger_vel']
        des_finger_torque[:1] += -K*l_error - D*state[0]['l_finger_vel']

        action = {'hand_control_type': 'position',
                  'des_hand_position': des_pose,
                  'grip_control_type': 'torque',
                  'des_grip_torque': des_finger_torque
                  }

        ## Set State Machine Transition
        error = pose_error.pow(2).sum(-1).pow(.5)
        if error<0.005:
            print('start grasp')
            self.state[idx] = 'grasp'
        return action

    def _squeeze(self, state, idx=0):
        ## Squeezing should achieve stable grasping / contact with the object of interest
        des_finger_torque = torch.ones(2)*-self.squeeze_force

        action = {'grip_control_type': 'torque',
                  'des_grip_torque': des_finger_torque}

        ## Set State Machine Transition after an empirical number of steps
        self.grasp_count[idx] +=1
        if self.grasp_count[idx]>300:
            self.grasp_count[idx] = 0
            self.grasp_state[idx] = 'hold'

        return action

    def _hold(self, state, idx=0):

        if self.grasp_count[idx] == 0:
            self.hold_force[idx] = self.squeeze_force
        else:
            self.hold_force[idx] +=1.0
        self.grasp_count[idx] += 1

        ## Set torques
        #print(self.hold_force[idx])
        des_finger_torque = torch.ones(2) * -self.hold_force[idx]
        action = {'grip_control_type': 'torque',
                  'des_grip_torque': des_finger_torque}

        ## Set State Machine Transition after an empirical number of steps, this also corresponded
        ## to increasing the desired grasping force for 100 steps
        if self.grasp_count[idx] > 100:
            #print(self.hold_force[idx])

            self.grasp_count[idx] = 0.
            self.l_finger_target[idx] = state['l_finger_pos'].clone()
            self.r_finger_target[idx] = state['r_finger_pos'].clone()

            self.state[idx] = 'lift'

        return action

    def _lift(self, state, idx=0):
        obj_pose = state['obj_pos']
        hand_pose  = state['hand_pos']
        #print('hand y: {}, obj y: {}'.format(hand_pose[1], obj_pose[1]))

        target_pose = torch.zeros_like(obj_pose)
        target_pose[2] = 2.
        target_pose[0] = 0.

        ## Set Desired Hand Pose
        pose_error = hand_pose - target_pose
        des_pose = hand_pose - pose_error*0.05

        ## Set Desired Grip Force
        des_finger_torque = torch.ones(2) * -self.hold_force[idx]

        r_error = state['r_finger_pos'] - self.r_finger_target[idx]
        l_error = state['l_finger_pos'] - self.l_finger_target[idx]

        K = 1000
        D = 20
        des_finger_torque[1:] += -K*r_error - D*state['r_finger_vel']
        des_finger_torque[:1] += -K*l_error - D*state['l_finger_vel']


        action = {'hand_control_type': 'position',
                  'des_hand_position': des_pose,
                  'grip_control_type': 'torque',
                  'des_grip_torque': des_finger_torque
                  }

        return action


class AnyIsaacGymWrapper():
    def __init__(self, env, viewer=True, physics='PHYSX', freq=250, device='cuda:0',
                 num_spaces=1, env_args=None, z_convention=False):

        self.franka_urdf = None

        ## Args
        self.sim_device_type, self.compute_device_id = gymutil.parse_device_str(device)
        self.device = device
        self.physics = physics
        self.freq = freq
        self.num_spaces = num_spaces
        self._set_transforms()
        self.z_convention = z_convention

        self.visualize = viewer

        ## Init Gym and Sim
        # @note 创建一个仿真世界，是指创建并初始化世界的参数等
        self.gym = gymapi.acquire_gym()
        self.sim, self.sim_params = self._create_sim()

        ## Create Visualizer
        if (self.visualize):
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                print("*** Failed to create viewer")
                quit()
        else:
            self.viewer = None

        ## Create Environment
        # @note 创建实际使用的仿真环境，是指地面、桌子、物体、夹爪等
        self._create_envs(env, env_args)

        ## Update camera pose
        if self.visualize:
            self._reset_camera(env_args)

    def _create_sim(self):
        """Set sim parameters and create a Sim object."""
        # Set simulation parameters

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / self.freq
        sim_params.gravity = gymapi.Vec3(0, -9.81, 0)
        sim_params.substeps = 1

        # Set stress visualization parameters
        sim_params.stress_visualization = True
        sim_params.stress_visualization_min = 1.0e2
        sim_params.stress_visualization_max = 1e5

        if self.z_convention:
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity = gymapi.Vec3(0., 0., -9.81)

        if self.physics == 'FLEX':
            sim_type = gymapi.SIM_FLEX
            print('using flex engine...')

            # Set FleX-specific parameters
            sim_params.flex.solver_type = 5
            sim_params.flex.num_outer_iterations = 10
            sim_params.flex.num_inner_iterations = 200
            sim_params.flex.relaxation = 0.75
            sim_params.flex.warm_start = 0.8

            sim_params.flex.deterministic_mode = True

            # Set contact parameters
            sim_params.flex.shape_collision_distance = 5e-4
            sim_params.flex.contact_regularization = 1.0e-6
            sim_params.flex.shape_collision_margin = 1.0e-4
            sim_params.flex.dynamic_friction = 0.7
        else:
            sim_type = gymapi.SIM_PHYSX
            print("using physx engine")
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 25
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.num_threads = 2
            sim_params.physx.use_gpu = True

            sim_params.physx.rest_offset = 0.0
            sim_params.physx.contact_offset = 0.001
            sim_params.physx.friction_offset_threshold = 0.001
            sim_params.physx.friction_correlation_distance = 0.0005

        # Create Sim object
        gpu_physics = self.compute_device_id
        if self.visualize:
            gpu_render = 0
        else:
            gpu_render = -1

        return self.gym.create_sim(gpu_physics, gpu_render, sim_type,
                                   sim_params), sim_params

    def _create_envs(self, env, env_args):
        # Add ground plane
        # 添加环境的地面
        plane_params = gymapi.PlaneParams()
        if self.z_convention:
            plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)  # 添加

        # Set up the env grid - only 1 object for now
        # 对应桌子的个数，一般对应同时运行的机器人数量
        num_envs = self.num_spaces
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # Some common handles for later use
        self.envs_gym = []
        self.envs = []
        print("Creating %d environments" % num_envs)
        num_per_row = int(math.sqrt(num_envs))   # 摆成一个正方形

        for i in range(num_envs):
            if isinstance(env_args, list):
                env_arg = env_args[i]
            else:
                env_arg = env_args

            # create env
            # @note 每个环境（桌子）对应一个句柄
            env_i = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs_gym.append(env_i)
            # now initialize the respective env:
            # @note 使用封装的 AnyGraspingGymEnv 对每个环境进行设置
            self.envs.append(env(self.gym, self.sim, env_i, self, i, env_arg))

    def _set_transforms(self):
        """Define transforms to convert between Trimesh and Isaac Gym conventions."""
        self.from_trimesh_transform = gymapi.Transform()
        self.from_trimesh_transform.r = gymapi.Quat(0, 0.7071068, 0,
                                                    0.7071068)
        self.neg_rot_x_transform = gymapi.Transform()
        self.neg_rot_x = gymapi.Quat(0.7071068, 0, 0, -0.7071068)
        self.neg_rot_x_transform.r = self.neg_rot_x

    def _reset_camera(self, args):
        if self.z_convention is False:
            # Point camera at environments
            cam_pos = gymapi.Vec3(0.0, 1.0, 0.6)
            cam_target = gymapi.Vec3(0.0, 0.8, 0.2)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        else:
            # Point camera at environments
            if args is not None:
                if 'cam_pose' in args:
                    cam = args['cam_pose']
                    cam_pos = gymapi.Vec3(cam[0], cam[1], cam[2])
                    cam_target = gymapi.Vec3(cam[3], cam[4], cam[5])
                else:
                    cam_pos = gymapi.Vec3(0.0, 0.9, 1.3)
                    cam_target = gymapi.Vec3(0.0, 0.0, .7)
            else:
                cam_pos = gymapi.Vec3(0.0, 0.9, 1.3)
                cam_target = gymapi.Vec3(0.0, 0.0, .7)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def get_franka_rpy(self, trimesh_grasp_quat):
        """Return RPY angles for Panda joints based on the grasp pose in the Z-up convention."""
        neg_rot_x = gymapi.Quat(0.7071068, 0, 0, -0.7071068)
        rot_z = gymapi.Quat(0, 0, 0.7071068, 0.7071068)
        desired_transform = neg_rot_x * trimesh_grasp_quat * rot_z
        r = R.from_quat([
            desired_transform.x, desired_transform.y, desired_transform.z,
            desired_transform.w
        ])
        return desired_transform, r.as_euler('ZYX')

    def reset(self, state=None):
        '''
        The reset function receives a list of dictionaries with the desired reset state for the different elements
        in the environment.
        '''
        for idx, env_i in enumerate(self.envs):
            if state is not None:
                env_i.reset(state[idx])
            else:
                env_i.reset()

        return self._evolve_step()

    def reset_robot(self, state=None, ensure_gripper_reset=False):
        '''
        The reset function receives a list of dictionaries with the desired reset state for the different elements
        in the environment. This function only resets the robot
        '''
        # if gripper reset should be ensured, we require two timesteps:
        if (ensure_gripper_reset):
            for idx, env_i in enumerate(self.envs):
                if state is not None:
                    env_i.reset_robot(state[idx],zero_grip_torque=True)
                else:
                    env_i.reset_robot(zero_grip_torque=True)
            self._evolve_step()


        for idx, env_i in enumerate(self.envs):
            if state is not None:
                env_i.reset_robot(state[idx])
            else:
                env_i.reset_robot()

        return self._evolve_step()

    def reset_obj(self, state=None):
        '''
        The reset function receives a list of dictionaries with the desired reset state for the different elements
        in the environment. This function only resets the robot
        '''
        for idx, env_i in enumerate(self.envs):
            if state is not None:
                env_i.reset_obj(state[idx])
            else:
                env_i.reset_obj()

        return self._evolve_step()

    def get_state(self):

        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rb_states = gymtorch.wrap_tensor(_rb_states)
        rb_states = rb_states.view(self.num_spaces, -1, rb_states.shape[-1])

        # DOF state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        n_dofs = self.envs[0].n_dofs
        dof_vel = dof_states[:, 1].view(self.num_spaces, n_dofs, 1)
        dof_pos = dof_states[:, 0].view(self.num_spaces, n_dofs, 1)

        s = []
        for idx, env_i in enumerate(self.envs):
            s.append(env_i.get_state([rb_states[idx,...], dof_pos[idx, ...], dof_vel[idx, ...]]))

        return s

    def step(self, action=None):

        if action is not None:
            for idx, env_i in enumerate(self.envs):
                env_i.step(action[idx])

        return self._evolve_step()

    def _evolve_step(self):
        # get the sim time
        t = self.gym.get_sim_time(self.sim)

        # Step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Step rendering
        self.gym.step_graphics(self.sim)

        if self.visualize:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)

        return self.get_state()

    def kill(self):
        self.gym.destroy_viewer(self.viewer)
        for env in self.envs_gym:
            self.gym.destroy_env(env)
        self.gym.destroy_sim(self.sim)
    
    def only_render(self):
        # Step rendering
        self.gym.step_graphics(self.sim)

        if self.visualize:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
        pass


class AnyGraspingGymEnv():
    """具体每个环境（桌子）的类
    """
    def __init__(self, gym, sim, env, isaac_base, curr_env_number, args=None):

        ## Set Args
        self.args = self._set_args(args)   # 环境的参数，即物体、夹爪等的参数
        self.n_dofs = 16

        ## Set Hyperparams
        self.gym = gym  # gymapi 的句柄
        self.sim = sim  # 仿真世界的句柄
        self.env = env  # 当前环境的句柄
        self.isaac_base = isaac_base             # 仿真世界封装类对象
        self.curr_env_number = curr_env_number   # 当前环境的编号

        ## Build Environment
        self._create_env()

    def _set_args(self, args):
        if args is None:
            obj_args = {
                'obj_mesh_path': 'rectangle',
                'scale': 1.,
            }
            args = {'obj_args':obj_args}
        else:
            args = args

        if 'obj_or' not in args['obj_args']:
            args['obj_args']['obj_or'] = np.array([0., 0., 0., 1.])

        return args

    def _create_env(self):
        # @note 核心：设置环境
        # 放置桌子
        self.table = self._load_table()
        # 放置物体
        self.obj, self.initial_obj_pose = self._load_obj(self.args['obj_args'])
        # 放置夹爪
        self.gripper = self._load_gripper(self.initial_obj_pose)

    def _load_table(self):
        # create table
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.armature = 0.01
        asset_options.use_mesh_materials = True
        table_path = TABLE_PATH
        table_asset = self.gym.load_asset(
            self.sim, '', table_path, asset_options)

        # table pose:
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, -0.02)
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_handle = self.gym.create_actor(self.env, table_asset, table_pose, "table", self.curr_env_number)
        return table_handle

    def _load_obj(self, args):
        obj_mesh_path = args['obj_mesh_path']
        obj_name = args['obj_name']
        scale = args['scale']

        #obj_ori = args['obj_ori']
        quat = args['obj_or']

        # create new shape object:
        obj_pose = gymapi.Transform()
        obj_pose.p = gymapi.Vec3(0.0, 0.0, 0.9)   # @note 物体的初始 translation
        obj_pose.r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])   # @note 物体的初始 rotation

        self.shape_obj = AnySimpleObject(self.gym, self.sim, self.env, self.isaac_base, self.curr_env_number, 
                                         obj_pose, 
                                         obj_mesh_path, obj_name,
                                         scale=scale)
        return self.shape_obj, obj_pose

    def _load_gripper(self, obj_pose):

        pose = gymapi.Transform()
        ## Compute initial rotation
        T_grasp = np.eye(4)
        Rot = R.from_euler('x', 180, degrees=True).as_matrix()
        T_grasp[:3, :3] = Rot
        grasp_trans = T_grasp[:3,-1]
        grasp_quat = R.from_matrix(T_grasp[:3,:3]).as_quat()
        ## Compute initial position
        pose.p = obj_pose.p
        pose.p.z += .6

        pose.r = gymapi.Quat(grasp_quat[0], grasp_quat[1],
                             grasp_quat[2], grasp_quat[3])

        gripper = AnyGripperOnly(self.gym, self.sim, self.env, self.isaac_base, self.curr_env_number, pose)
        self.initial_Hgrip = Transform_2_H(pose)
        return gripper

    def get_state(self, rb_states=None):
        if rb_states is None:
            _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
            rb_states = gymtorch.wrap_tensor(_rb_states)

        rb_state = rb_states[0]
        dof_pos  = rb_states[1]
        dof_vel  = rb_states[2]

        ## get object state
        obj_state = self.obj.get_state(rb_state)

        hand_state = self.gripper.get_state(rb_state, dof_pos = dof_pos, dof_vel = dof_vel)

        return {**hand_state, **obj_state}

    def step(self, a=None):
        self.gripper.set_action(a)

    def reset(self, state_dict={}):
        # 设置物体的位姿
        if 'obj_state' in state_dict:
            self.obj.reset(state_dict['obj_state'])
        else:
            self.obj.reset(self.initial_obj_pose)

        # @note 设置夹爪的位姿
        if 'grip_state' in state_dict:
            self.gripper.reset(state_dict['grip_state'])
        else:
            self.gripper.reset(self.initial_Hgrip)


class AnySimpleObject():
    def __init__(self, gym, sim, env, isaac_base, env_number, 
                pose, 
                obj_mesh_path, obj_name, 
                args = None,
                collision_group=1, segmentationId=0, linearDamping=0, angularDamping=0, scale=1., disable_gravity=True):

        ##Set arguments
        self.args = self._set_args(args)
        self.disable_gravity = disable_gravity

        ## Set Hyperparameters
        # @note 老样子，把所有的句柄都传进来
        self.gym = gym
        self.sim = sim
        self.env = env
        self.isaac_base = isaac_base

        # 物体的初始位姿
        self.initial_pose = copy.deepcopy(pose)

        ##Set args
        self.obj_mesh_path = obj_mesh_path

        self.linearDamping = linearDamping
        self.angularDamping = angularDamping

        ## Set assets
        # @note 新建一个物体，并初始化
        obj_assets = self._set_assets()
        self.handle = gym.create_actor(env, obj_assets, pose, obj_name,
                                       group=env_number, filter=collision_group, segmentationId=segmentationId)
        # print('Object Handle: {}'.format(self.handle))
        self.gym.set_actor_scale(self.env, self.handle, scale)
        pass

    def _set_args(self, args):
        if args is None:
            args ={
                'physics':'PHYSX',
            }
        else:
            args = args
        return args

    def _get_objs_path(self):
        # @note obj 文件转换为 urdf 文件的函数
        res_urdf_path = generate_obj_urdf(self.obj_mesh_path)

        return res_urdf_path

    def _set_assets(self):
        asset_file_object = self._get_objs_path()

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False

        #asset_options.flip_visual_attachments = False
        asset_options.armature = 0.
        asset_options.thickness = 0.
        asset_options.density = 1000.

        asset_options.linear_damping = self.linearDamping  # Linear damping for rigid bodies
        asset_options.angular_damping = self.angularDamping  # Angular damping for rigid bodies
        asset_options.disable_gravity = self.disable_gravity
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 200000

        # @note 加载模型文件
        obj_asset = self.gym.load_asset(
            self.sim, '', asset_file_object, asset_options)
        return obj_asset

    def get_state(self, rb_states=None):
        if rb_states is None:
            _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
            rb_states = gymtorch.wrap_tensor(_rb_states)

        obj_state = rb_states[self.handle,...]
        obj_pos = obj_state[:3]
        obj_rot = obj_state[3:7]
        obj_vel = obj_state[7:]

        H = pq_to_H(obj_pos, obj_rot)

        return {'obj_pos':obj_pos, 'obj_rot':obj_rot, 'obj_vel': obj_vel, 'H_obj':H}

    def get_rigid_body_state(self):
        # gets state of exactly this rigid body
        return self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_ALL)

    def reset(self, H):
        pos = [H.p.x, H.p.y, H.p.z]
        rot = [H.r.x, H.r.y, H.r.z, H.r.w]
        self.set_rigid_body_pos(pos, rot)

    def set_rigid_body_pos(self, pos, ori):
        # sets the position of the ridgid body and the velocity to zero
        obj = self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_NONE)
        obj['pose']['p'].fill((pos[0],pos[1],pos[2]))
        obj['pose']['r'].fill((ori[0],ori[1],ori[2],ori[3]))
        obj['vel']['linear'].fill((0,0,0))
        obj['vel']['angular'].fill((0,0,0))
        self.gym.set_actor_rigid_body_states(self.env, self.handle, obj, gymapi.STATE_ALL)

    def set_rigid_body_pos_keep_vel(self, pos, ori):
        # sets the position of the ridgid body and keeps the velocity
        obj = self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_ALL)
        obj['pose']['p'].fill((pos[0],pos[1],pos[2]))
        obj['pose']['r'].fill((ori[0],ori[1],ori[2],ori[3]))
        self.gym.set_actor_rigid_body_states(self.env, self.handle, obj, gymapi.STATE_ALL)

    def set_rigid_body_pos_vel(self, pos, ori, vel_lin, vel_ang):
        # sets the position and velocity
        obj = self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_NONE)
        obj['pose']['p'].fill((pos[0],pos[1],pos[2]))
        obj['pose']['r'].fill((ori[0],ori[1],ori[2],ori[3]))
        obj['vel']['linear'].fill((vel_lin[0],vel_lin[1],vel_lin[2]))
        obj['vel']['angular'].fill((vel_ang[0],vel_ang[1],vel_ang[2]))
        self.gym.set_actor_rigid_body_states(self.env, self.handle, obj, gymapi.STATE_ALL)
    

class AnyGripperOnly():
    def __init__(self, gym, sim, env, isaac_base, env_number, pose, collision_group=0, segmentationId=0):

        ## Hyperparameters
        self.gym = gym
        self.sim = sim
        self.env = env
        self.isaac_base = isaac_base


        ## Controller Args
        self.hand_cntrl_type    = 'position'
        self.grip_cntrl_type = 'torque'

        ## State args
        self.base_pose = pose

        ## Set assets
        self.franka_asset = self.set_assets()

        # add the franka hand
        self.handle = self.gym.create_actor(self.env, self.franka_asset, pose, "franka", group=env_number, filter=collision_group, segmentationId=segmentationId)
        self.hand_handle = self.gym.find_actor_rigid_body_handle(self.env, self.handle, "panda_hand")
        self.base_handle = self.gym.find_actor_rigid_body_handle(self.env, self.handle, "world")

        ## Set initial joint configuration ##
        curr_joint_positions = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_ALL)
        curr_joint_positions['pos'][-1] = 0.04
        curr_joint_positions['pos'][-2] = 0.04

        self.gym.set_actor_dof_states(self.env, self.handle,
                                      curr_joint_positions, gymapi.STATE_ALL)

        curr_joint_positions = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_ALL)


        self.gym.set_actor_dof_position_targets(self.env, self.handle, curr_joint_positions['pos'])

        ## Set control properties
        self._set_cntrl_properties()
        self.target_pose = curr_joint_positions['pos']
        self.target_torque = torch.zeros(self.target_pose.shape[0])

        # Attractor

        self._set_initial_target = True

    def set_assets(self):
        # Load franka asset
        franka_asset_file = FRANKA_URDF_PATH
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.armature = 0.0
        asset_options.thickness = 0.0

        asset_options.linear_damping = 100.0  # Linear damping for rigid bodies
        asset_options.angular_damping = 100.0  # Angular damping for rigid bodies
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_VEL
        asset_options.fix_base_link = True
        asset_options.vhacd_enabled = True

        return self.gym.load_asset(
            self.sim, '', franka_asset_file, asset_options)

    def get_state(self, rb_states, dof_pos, dof_vel):

        ## Base state
        base_state = rb_states[self.base_handle, ...]
        base_pos = base_state[:3]
        base_rot = base_state[3:7]
        base_vel_p = base_state[7:10]
        base_vel_r = base_state[10:]

        self.base_state_dict = {'base_pos': base_pos, 'base_rot': base_rot,
                           'base_vel_p': base_vel_p, 'base_vel_r': base_vel_r}

        ## Hand state
        hand_state = rb_states[self.hand_handle, ...]
        hand_pos = hand_state[:3]
        hand_rot = hand_state[3:7]
        #print('pos: {}, ori:{}'.format(hand_pos, hand_rot))

        hand_vel_p = hand_state[7:10]
        hand_vel_r = hand_state[10:]


        hand_state_dict = {'hand_pos': hand_pos, 'hand_rot': hand_rot,
                           'hand_vel_p': hand_vel_p, 'hand_vel_r': hand_vel_r}

        self.robot_state = [dof_pos, dof_vel]
        ## Fingers state
        finger_state_dict = {'r_finger_pos': dof_pos[-1], 'l_finger_pos': dof_pos[-2],
                             'r_finger_vel': dof_vel[-1], 'l_finger_vel': dof_vel[-2],
                             'position_pos': dof_pos[:6]}


        return {**hand_state_dict, **finger_state_dict}

    def _init_cntrl(self):
        attractor_properties = gymapi.AttractorProperties()
        attractor_properties.stiffness = 5e5
        attractor_properties.damping = 5e3
        attractor_properties.axes = gymapi.AXIS_ALL
        attractor_properties.rigid_handle = self.hand_handle

        self.attractor_handle = self.gym.create_rigid_body_attractor(self.env, attractor_properties)
        hand_pose = self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_POS)['pose'][:][-4]

        #print('Target pose: {}'.format(hand_pose))

        self.gym.set_attractor_target(self.env, self.attractor_handle, hand_pose)

    def _set_cntrl_properties(self, grip_cntrl_type='torque'):
        # get joint limits and ranges for Franka
        franka_dof_props = self.gym.get_asset_dof_properties(self.franka_asset)
        franka_lower_limits = franka_dof_props['lower']
        franka_upper_limits = franka_dof_props['upper']
        franka_ranges = franka_upper_limits - franka_lower_limits
        franka_mids = 0.5 * (franka_upper_limits + franka_lower_limits)
        franka_num_dofs = len(franka_dof_props)

        # set DOF control properties (except grippers)
        franka_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][:].fill(400.0)
        franka_dof_props["damping"][:].fill(100.0)

        if grip_cntrl_type =='torque':
            self.grip_cntrl_type = grip_cntrl_type
            # set DOF control properties for grippers
            franka_dof_props["driveMode"][-2:].fill(gymapi.DOF_MODE_EFFORT)
            franka_dof_props["stiffness"][-2:].fill(0.0)
            franka_dof_props["damping"][-2:].fill(0.0)
        else:
            self.grip_cntrl_type = 'position'
            franka_dof_props["driveMode"][-2:].fill(gymapi.DOF_MODE_POS)
            franka_dof_props["stiffness"][-2:].fill(200.0)
            franka_dof_props["damping"][-2:].fill(40.0)

        # Set DOF control properties
        self.gym.set_actor_dof_properties(self.env, self.handle, franka_dof_props)

    def set_action(self, action):

        ## Set initial attractor's target
        if self._set_initial_target:
            self._set_initial_target = False
            self._init_cntrl()

        if 'hand_control_type' in  action:

            if action['hand_control_type']=='position':
                ## Set desired position displacement
                hand_xyz = action['des_hand_position']

                # The attractor is used to move the hand
                attractor_properties = self.gym.get_attractor_properties(self.env, self.attractor_handle)
                pose = attractor_properties.target
                # print('target pos: ({}, {}, {}), target ori: ({} {} {} {})'.format(pose.p.x, pose.p.y, pose.p.z,
                #                                                                    pose.r.x, pose.r.y, pose.r.z, pose.r.w))
                pose.p.x = hand_xyz[0]
                pose.p.y = hand_xyz[1]
                pose.p.z = hand_xyz[2]
                self.gym.set_attractor_target(self.env, self.attractor_handle, pose)


        if 'grip_control_type' in  action:

            if action['grip_control_type'] != self.grip_cntrl_type:
                self._set_cntrl_properties(grip_cntrl_type=action['grip_control_type'])

            if action['grip_control_type']=='torque':
                torque_grip = action['des_grip_torque']
                self.target_torque[-2:] = torque_grip

        ## Set controllers ##
        self.gym.apply_actor_dof_efforts(self.env, self.handle, self.target_torque)

    def reset(self, H):
        ## Set root
        T = H_2_Transform(H)
        self.gym.set_rigid_transform(self.env, self.handle, T)

        ## Set DOF to zero
        curr_joint_positions = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_ALL)
        #print(curr_joint_positions)
        curr_joint_positions['pos'] = np.zeros_like(curr_joint_positions['pos'])
        curr_joint_positions['vel'] = np.zeros_like(curr_joint_positions['vel'])
        curr_joint_positions['pos'][-1] = 0.04
        curr_joint_positions['pos'][-2] = 0.04

        self.gym.set_actor_dof_states(self.env, self.handle,
                                      curr_joint_positions, gymapi.STATE_ALL)

        self.gym.set_actor_dof_position_targets(self.env, self.handle, curr_joint_positions['pos'])

        if self._set_initial_target:
             self._set_initial_target = False
             self._init_cntrl()
        else:
            hand_pose = self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_POS)['pose'][:][-4]
            self.gym.set_attractor_target(self.env, self.attractor_handle, hand_pose)

            self.target_torque = torch.zeros_like(self.target_torque)
            self.gym.apply_actor_dof_efforts(self.env, self.handle, self.target_torque)
