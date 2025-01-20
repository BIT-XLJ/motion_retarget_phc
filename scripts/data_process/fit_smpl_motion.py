import glob
import os
import sys
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from smpl_sim.smpllib.smpl_parser import SMPL_Parser
import joblib
from tqdm import tqdm
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES
from easydict import EasyDict
import hydra
from omegaconf import DictConfig, OmegaConf
import casadi                                                                       
import meshcat.geometry as mg
import pinocchio as pin   
from pinocchio import casadi as cpin                   
from pinocchio.visualize import MeshcatVisualizer  
from weighted_moving_filter import WeightedMovingFilter

sys.path.append(os.getcwd())
parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)

def rotx(theta):
    theta = np.radians(theta) 
    Rx = np.array([[1, 0, 0],
               [0, np.cos(theta), -np.sin(theta)],
               [0, np.sin(theta), np.cos(theta)]])
    return Rx
    
def roty(theta):
    theta = np.radians(theta) 
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
               [0, 1, 0],
               [-np.sin(theta), 0, np.cos(theta)]])
    return Ry
    
def rotz(theta):
    theta = np.radians(theta)
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
               [np.sin(theta), np.cos(theta), 0],
               [0, 0, 1]])
    return Rz

def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    if not 'mocap_framerate' in  entry_data:
        return 
    framerate = entry_data['mocap_framerate']

    root_trans = entry_data['trans']
    pose_aa = np.concatenate([entry_data['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis = -1)
    betas = entry_data['betas']
    gender = entry_data['gender']
    N = pose_aa.shape[0]
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans, 
        "betas": betas,
        "fps": framerate
    }
    
def process_motion(key_names, key_name_to_pkls, cfg):
    
    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")
    shape_new, scale = joblib.load(f"data/{cfg.robot.humanoid_type}/shape_optimized_v1.pkl") # TODO: run fit_smple_shape to get this
    
    all_data = {}
    pbar = tqdm(key_names, position=0, leave=True)
    for data_key in pbar:
        print("key_name_to_pkls----------------------")
        print(key_name_to_pkls)
        print("--------------------------------------")
        amass_data = load_amass_data(key_name_to_pkls[data_key])
        if amass_data is None: continue
        skip = int(amass_data['fps']//60)
        trans = torch.from_numpy(amass_data['trans'][::skip])  #::skip是切片，
        pose_aa_walk = torch.from_numpy(amass_data['pose_aa'][::skip]).float()
        
        verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
        root_pos = joints[:, 0:1] #joints代表的是关节的位置数据
        joints = (joints - joints[:, 0:1]) * scale.detach() + root_pos
        joints[..., 2] = joints[..., 2] - verts[0, :, 2].min().item() - 0.7
        
        N = joints.shape[0]
        link_homogeneous_tf = np.zeros((np.int32(pose_aa_walk.shape[1]/3), N, 4, 4))
        for i in range(np.int32(pose_aa_walk.shape[1]/3)):
            rotation_matrices = sRot.from_rotvec(pose_aa_walk[:, i:i+3]).as_matrix()
            homogeneous_matrices = np.zeros((N, 4, 4))
            
            if i == 0:
                root_matrix = rotation_matrices @ rotx(-90) @ rotz(-90)
                rot = sRot.from_matrix(root_matrix)
                euler_angles = rot.as_euler('ZYX', degrees=False)
                yaw_only_euler = np.zeros_like(euler_angles)
                yaw_only_euler[:, 0] = euler_angles[:, 0]
                yaw_only_rotation = sRot.from_euler('ZYX', yaw_only_euler, degrees=False)
                yaw_only_matrices = yaw_only_rotation.as_matrix()
                homogeneous_matrices[:, :3, :3] = yaw_only_matrices   #only consider about yaw
            else:
                homogeneous_matrices[:, :3, :3] = root_matrix @ rotation_matrices
                
            homogeneous_matrices[:, :3, 3] = joints[:, i, :]
            homogeneous_matrices[:, 3, 3] = 1
            link_homogeneous_tf[i,:,:,:] = homogeneous_matrices
            # print(rotation_matrices)
        
        all_data[data_key] = {
            "link_homogeneous_tf": link_homogeneous_tf,
        }
            
    return all_data        
        

    
class RobotIK:
    def __init__(self, Visualization = False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Visualization = Visualization
        
        self.joint_model = pin.JointModelComposite()
        # 添加 3 个平移自由度
        self.joint_model.addJoint(pin.JointModelTranslation())

        # 添加 3 个旋转自由度 (roll, pitch, yaw)
        self.joint_model.addJoint(pin.JointModelRX())  # Roll
        self.joint_model.addJoint(pin.JointModelRY())  # Pitch
        self.joint_model.addJoint(pin.JointModelRZ())  # Yaw

        current_path = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(current_path, '../../phc/data/assets/robot/noetix_n2/urdf', 'N2.urdf')
        urdf_dirs = os.path.join(current_path, '../../phc/data/assets/robot/noetix_n2/urdf')
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path,
                                                    root_joint = self.joint_model,
                                                    package_dirs = urdf_dirs)

        self.mixed_jointsToLockIDs =[]
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )
        
        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()
        # print("输出pinocchio关节顺序")
        # print(self.cmodel)
        
        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_lhand = casadi.SX.sym("tf_lhand", 4, 4) #only consider about transform, not included rotation
        self.cTf_rhand = casadi.SX.sym("tf_rhand", 4, 4)
        self.cTf_lelbow = casadi.SX.sym("tf_lelbow", 4, 4)
        self.cTf_relbow = casadi.SX.sym("tf_relbow", 4, 4)            
        self.cTf_root  = casadi.SX.sym("tf_root" , 4, 4)
        self.cTf_lfoot = casadi.SX.sym("tf_lfoot", 4, 4)
        self.cTf_rfoot = casadi.SX.sym("tf_rfoot", 4, 4)
        self.cTf_lknee = casadi.SX.sym("tf_lknee", 4, 4)
        self.cTf_rknee = casadi.SX.sym("tf_rknee", 4, 4)
           
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.lhand_id = self.reduced_robot.model.getFrameId("L_arm_hand_Link")
        self.rhand_id = self.reduced_robot.model.getFrameId("R_arm_hand_Link")
        self.lelbow_id = self.reduced_robot.model.getFrameId("L_arm_elbow_Link")
        self.relbow_id = self.reduced_robot.model.getFrameId("R_arm_elbow_Link")        
        self.root_id = self.reduced_robot.model.getFrameId("base_link") # pelvis
        self.lfoot_id = self.reduced_robot.model.getFrameId("L_leg_ankle_link")
        self.rfoot_id = self.reduced_robot.model.getFrameId("R_leg_ankle_link")
        self.lknee_id = self.reduced_robot.model.getFrameId("L_leg_knee_link")
        self.rknee_id = self.reduced_robot.model.getFrameId("R_leg_knee_link")        
        
        # self.translational_error = casadi.Function(
        #     "translational_error",
        #     [self.cq, self.cTf_lhand, self.cTf_rhand, self.cTf_lelbow, self.cTf_relbow, self.cTf_root, self.cTf_lfoot, self.cTf_rfoot, self.cTf_lknee, self.cTf_rknee],
        #     [
        #         casadi.vertcat(
        #             self.cdata.oMf[self.lhand_id].translation - self.cTf_lhand[:3,3],
        #             self.cdata.oMf[self.rhand_id].translation - self.cTf_rhand[:3,3],
        #             self.cdata.oMf[self.lelbow_id].translation - self.cTf_lelbow[:3,3],
        #             self.cdata.oMf[self.relbow_id].translation - self.cTf_relbow[:3,3],  
        #             self.cdata.oMf[self.root_id].translation - self.cTf_root[:3,3],
        #             self.cdata.oMf[self.lfoot_id].translation - self.cTf_lfoot[:3,3],
        #             self.cdata.oMf[self.rfoot_id].translation - self.cTf_rfoot[:3,3],
        #             self.cdata.oMf[self.lknee_id].translation - self.cTf_lknee[:3,3],
        #             self.cdata.oMf[self.rknee_id].translation - self.cTf_rknee[:3,3],                    
        #         )
        #     ],
        # )
        
        self.hand_translational_error = casadi.Function(
            "hand_translational_error",
            [self.cq, self.cTf_lhand, self.cTf_rhand],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.lhand_id].translation - self.cTf_lhand[:3,3],
                    self.cdata.oMf[self.rhand_id].translation - self.cTf_rhand[:3,3],                   
                )
            ],
        )        

        self.elbow_translational_error = casadi.Function(
            "elbow_translational_error",
            [self.cq, self.cTf_lelbow, self.cTf_relbow],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.lelbow_id].translation - self.cTf_lelbow[:3,3],
                    self.cdata.oMf[self.relbow_id].translation - self.cTf_relbow[:3,3],                   
                )
            ],
        )

        self.root_translational_error = casadi.Function(
            "root_translational_error",
            [self.cq, self.cTf_root],
            [
                casadi.vertcat( 
                    self.cdata.oMf[self.root_id].translation - self.cTf_root[:3,3],                  
                )
            ],
        )
        
        self.foot_translational_error = casadi.Function(
            "foot_translational_error",
            [self.cq, self.cTf_lfoot, self.cTf_rfoot],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.lfoot_id].translation - self.cTf_lfoot[:3,3],
                    self.cdata.oMf[self.rfoot_id].translation - self.cTf_rfoot[:3,3],                   
                )
            ],
        )        
        
        self.knee_translational_error = casadi.Function(
            "knee_translational_error",
            [self.cq, self.cTf_lknee, self.cTf_rknee],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.lknee_id].translation - self.cTf_lknee[:3,3],
                    self.cdata.oMf[self.rknee_id].translation - self.cTf_rknee[:3,3],                    
                )
            ],
        )

                
        # self.rotational_error = casadi.Function(
        #     "rotational_error",
        #     [self.cq, self.cTf_lhand, self.cTf_rhand, self.cTf_lelbow, self.cTf_relbow, self.cTf_root, self.cTf_lfoot, self.cTf_rfoot, self.cTf_lknee, self.cTf_rknee],
        #     [
        #         casadi.vertcat(
        #             cpin.log3(self.cdata.oMf[self.lhand_id].rotation @ self.cTf_lhand[:3,:3].T),
        #             cpin.log3(self.cdata.oMf[self.rhand_id].rotation @ self.cTf_rhand[:3,:3].T),
        #             cpin.log3(self.cdata.oMf[self.lelbow_id].rotation @ self.cTf_lelbow[:3,:3].T),
        #             cpin.log3(self.cdata.oMf[self.relbow_id].rotation @ self.cTf_relbow[:3,:3].T),   
        #             cpin.log3(self.cdata.oMf[self.root_id].rotation @ self.cTf_root[:3,:3].T),
        #             cpin.log3(self.cdata.oMf[self.lfoot_id].rotation @ self.cTf_lfoot[:3,:3].T),
        #             cpin.log3(self.cdata.oMf[self.rfoot_id].rotation @ self.cTf_rfoot[:3,:3].T),  
        #             cpin.log3(self.cdata.oMf[self.lknee_id].rotation @ self.cTf_lknee[:3,:3].T),
        #             cpin.log3(self.cdata.oMf[self.rknee_id].rotation @ self.cTf_rknee[:3,:3].T),                                                           
        #         )
        #     ],
        # )

        self.hand_rotational_error = casadi.Function(
            "hand_rotational_error",
            [self.cq, self.cTf_lhand, self.cTf_rhand],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.lhand_id].rotation @ self.cTf_lhand[:3,:3].T),
                    cpin.log3(self.cdata.oMf[self.rhand_id].rotation @ self.cTf_rhand[:3,:3].T),                                                          
                )
            ],
        )
        
        self.elbow_rotational_error = casadi.Function(
            "elbow_rotational_error",
            [self.cq, self.cTf_lelbow, self.cTf_relbow],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.lelbow_id].rotation @ self.cTf_lelbow[:3,:3].T),
                    cpin.log3(self.cdata.oMf[self.relbow_id].rotation @ self.cTf_relbow[:3,:3].T),                                                            
                )
            ],
        )        

        self.root_rotational_error = casadi.Function(
            "root_rotational_error",
            [self.cq,  self.cTf_root],
            [
                casadi.vertcat(  
                    cpin.log3(self.cdata.oMf[self.root_id].rotation @ self.cTf_root[:3,:3].T),                                                          
                )
            ],
        )
        
        self.foot_rotational_error = casadi.Function(
            "foot_rotational_error",
            [self.cq, self.cTf_lfoot, self.cTf_rfoot],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.lfoot_id].rotation @ self.cTf_lfoot[:3,:3].T),
                    cpin.log3(self.cdata.oMf[self.rfoot_id].rotation @ self.cTf_rfoot[:3,:3].T),                                                           
                )
            ],
        )
        
        self.knee_rotational_error = casadi.Function(
            "knee_rotational_error",
            [self.cq, self.cTf_lknee, self.cTf_rknee],
            [
                casadi.vertcat(  
                    cpin.log3(self.cdata.oMf[self.lknee_id].rotation @ self.cTf_lknee[:3,:3].T),
                    cpin.log3(self.cdata.oMf[self.rknee_id].rotation @ self.cTf_rknee[:3,:3].T),                                                           
                )
            ],
        )        
                
        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        
        self.param_tf_lhand = self.opti.parameter(4, 4)
        self.param_tf_rhand = self.opti.parameter(4, 4)
        self.param_tf_lelbow = self.opti.parameter(4, 4)
        self.param_tf_relbow = self.opti.parameter(4, 4)        
        self.param_tf_root = self.opti.parameter(4, 4)
        self.param_tf_lfoot = self.opti.parameter(4, 4)
        self.param_tf_rfoot = self.opti.parameter(4, 4)
        self.param_tf_lknee = self.opti.parameter(4, 4)
        self.param_tf_rknee = self.opti.parameter(4, 4)
                
        self.hand_translational_cost = casadi.sumsqr(self.hand_translational_error(self.var_q, self.param_tf_lhand, self.param_tf_rhand))
        self.elbow_translational_cost = casadi.sumsqr(self.elbow_translational_error(self.var_q, self.param_tf_lelbow, self.param_tf_relbow))
        self.root_translational_cost = casadi.sumsqr(self.root_translational_error(self.var_q, self.param_tf_root))
        self.foot_translational_cost = casadi.sumsqr(self.foot_translational_error(self.var_q, self.param_tf_lfoot, self.param_tf_rfoot))
        self.knee_translational_cost = casadi.sumsqr(self.knee_translational_error(self.var_q, self.param_tf_lknee, self.param_tf_rknee))
                     
        self.hand_rotation_cost = casadi.sumsqr(self.hand_rotational_error(self.var_q, self.param_tf_lhand, self.param_tf_rhand))
        self.elbow_rotation_cost = casadi.sumsqr(self.elbow_rotational_error(self.var_q, self.param_tf_lelbow, self.param_tf_relbow))
        self.root_rotation_cost = casadi.sumsqr(self.root_rotational_error(self.var_q, self.param_tf_root))
        self.foot_rotation_cost = casadi.sumsqr(self.foot_rotational_error(self.var_q, self.param_tf_lfoot, self.param_tf_rfoot))
        self.knee_rotation_cost = casadi.sumsqr(self.knee_rotational_error(self.var_q, self.param_tf_lknee, self.param_tf_rknee))
                
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        self.opti.minimize( 20 * self.root_translational_cost + 50 * self.foot_translational_cost + 20 *self.knee_translational_cost + 6 * self.elbow_translational_cost + 12 * self.hand_translational_cost + \
                            1 * self.root_rotation_cost + 0.5 * self.foot_rotation_cost + 1 * self.knee_rotation_cost + 0.1*self.elbow_rotation_cost + 0.05*self.hand_rotation_cost + 0.02 * self.regularization_cost + 0.8 * self.smooth_cost )
        
        opts = {
            'ipopt':{
                'print_level':0,
                'max_iter':50,
                'tol':1e-3
            },
            'print_time':False,# print or not
            'calc_lam_p':False 
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), self.reduced_robot.model.nq)
        self.vis = None

        if self.Visualization:
            # Initialize the Meshcat visualizer for visualization
            self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
            self.vis.initViewer(open=True) 
            self.vis.loadViewerModel("pinocchio") 
            self.vis.displayFrames(True, frame_ids=[101, 102], axis_length = 0.15, axis_width = 5)
            self.vis.display(pin.neutral(self.reduced_robot.model))

            # Enable the display of end effector target frames with short axis lengths and greater width.
            frame_viz_names = ['lhand_target', 'rhand_target', 'lfoot_target', 'rfoot_target', 'root_target', 'lelbow_target', 'relbow_target', 'lknee_target', 'rknee_target']
            FRAME_AXIS_POSITIONS = (
                np.array([[0, 0, 0], [1, 0, 0],
                          [0, 0, 0], [0, 1, 0],
                          [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
            )
            FRAME_AXIS_COLORS = (
                np.array([[1, 0, 0], [1, 0.6, 0],
                          [0, 1, 0], [0.6, 1, 0],
                          [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
            )
            axis_length = 0.1
            axis_width = 10
            for frame_viz_name in frame_viz_names:
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(
                            position=axis_length * FRAME_AXIS_POSITIONS,
                            color=FRAME_AXIS_COLORS,
                        ),
                        mg.LineBasicMaterial(
                            linewidth=axis_width,
                            vertexColors=True,
                        ),
                    )
                )

    def solve_ik(self, left_hand, right_hand, left_elbow, right_elbow, root, left_foot, right_foot, left_knee, right_knee, current_lr_arm_motor_q = None, current_lr_arm_motor_dq = None):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        if self.Visualization:
            self.vis.viewer['lhand_target'].set_transform(left_hand)          # for visualization
            self.vis.viewer['rhand_target'].set_transform(right_hand)  # for visualization
            self.vis.viewer['lfoot_target'].set_transform(left_foot)   # for visualization
            self.vis.viewer['rfoot_target'].set_transform(right_foot)  # for visualization
            self.vis.viewer['root_target'].set_transform(root)         # for visualization
            self.vis.viewer['lelbow_target'].set_transform(left_elbow)       # for visualization
            self.vis.viewer['relbow_target'].set_transform(right_elbow)       # for visualization
            self.vis.viewer['lknee_target'].set_transform(left_knee)        # for visualization
            self.vis.viewer['rknee_target'].set_transform(right_knee)        # for visualization

        self.opti.set_value(self.param_tf_lhand, left_hand)
        self.opti.set_value(self.param_tf_rhand, right_hand)
        self.opti.set_value(self.param_tf_lfoot, left_foot)
        self.opti.set_value(self.param_tf_rfoot, right_foot)
        self.opti.set_value(self.param_tf_root, root)
        self.opti.set_value(self.param_tf_lelbow, left_elbow)
        self.opti.set_value(self.param_tf_relbow, right_elbow)
        self.opti.set_value(self.param_tf_lknee, left_knee)
        self.opti.set_value(self.param_tf_rknee, right_knee)

        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            sol = self.opti.solve()
            sol_q = self.opti.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0
            self.init_data = sol_q
            if self.Visualization:
                self.vis.display(sol_q)  # for visualization
            return sol_q

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")
            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data
            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0
            self.init_data = sol_q
            print(f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \nleft_pose: \n{left_hand} \nright_pose: \n{right_hand}")
            if self.Visualization:
                self.vis.display(sol_q)  # for visualization
            # return sol_q, sol_tauff
            return current_lr_arm_motor_q        


@hydra.main(version_base=None, config_path="../../phc/data/cfg", config_name="config")
def main(cfg : DictConfig) -> None:
    if "amass_root" in cfg:
        amass_root = cfg.amass_root
    else:
        raise ValueError("amass_root is not specified in the config")
    
    all_pkls = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    split_len = len(amass_root.split("/"))
    key_name_to_pkls = {"0-" + "_".join(data_path.split("/")[split_len:]).replace(".npz", ""): data_path for data_path in all_pkls}
    key_names = ["0-" + "_".join(data_path.split("/")[split_len:]).replace(".npz", "") for data_path in all_pkls]
    print("key_names-------------------")
    print(key_names)
    print("---------------------------")
    
    from multiprocessing import Pool
    jobs = key_names
    num_jobs = 30
    chunk = np.ceil(len(jobs)/num_jobs).astype(int)
    jobs= [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
    job_args = [(jobs[i], key_name_to_pkls, cfg) for i in range(len(jobs))]
    if len(job_args) == 1:
        all_data = process_motion(key_names, key_name_to_pkls, cfg)
    else:
        try:
            pool = Pool(num_jobs)   # multi-processing
            all_data_list = pool.starmap(process_motion, job_args)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
        all_data = {}
        for data_dict in all_data_list:
            all_data.update(data_dict)
    
    if len(all_data) == 1:
        data_key = list(all_data.keys())[0]
        os.makedirs(f"data/{cfg.robot.humanoid_type}/v1/singles", exist_ok=True)
        dumped_file = f"data/{cfg.robot.humanoid_type}/v1/singles/{data_key}.pkl"
        print(dumped_file)
        joblib.dump(all_data, dumped_file)
    else:
        os.makedirs(f"data/{cfg.robot.humanoid_type}/v1/", exist_ok=True)
        joblib.dump(all_data, f"data/{cfg.robot.humanoid_type}/v1/amass_all.pkl")
     
    smpl_joint_pick = [i[1] for i in cfg.robot.smpl_joints_matches]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]
    
    
    robot_ik = RobotIK(Visualization = True)
    

    for data_key, data_value in all_data.items():
        joints_tf = data_value["link_homogeneous_tf"]

    # print(joints_pos[0,smpl_joint_pick_idx[0]].view(3,1).numpy())
    # print(root_tf.shape)
    sol_q_last = np.zeros(24)
    sol_q_last[12] = -0.1        #赋一个初始值,注意这里的顺序要和pinocchio动力学库的关节顺序保持一致
    sol_q_last[13] = 0.3
    sol_q_last[14] = -0.2
    sol_q_last[21] = -0.1
    sol_q_last[22] = 0.3
    sol_q_last[23] = -0.2
    
    motion_num = joints_tf.shape[1]
    print(motion_num)
    
    
    
    for i in range(motion_num):
        sol_q = robot_ik.solve_ik(joints_tf[smpl_joint_pick_idx[0],i],
                                  joints_tf[smpl_joint_pick_idx[1],i],
                                  joints_tf[smpl_joint_pick_idx[2],i],
                                  joints_tf[smpl_joint_pick_idx[3],i],
                                  joints_tf[smpl_joint_pick_idx[4],i],
                                  joints_tf[smpl_joint_pick_idx[5],i],
                                  joints_tf[smpl_joint_pick_idx[6],i],
                                  joints_tf[smpl_joint_pick_idx[7],i],
                                  joints_tf[smpl_joint_pick_idx[8],i],
                                  current_lr_arm_motor_q=sol_q_last
                                )
        sol_q_last = sol_q
        
if __name__ == "__main__":
    main()
