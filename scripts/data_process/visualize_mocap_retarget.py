import meshcat.geometry as mg
import pinocchio as pin   
from pinocchio import casadi as cpin                   
from pinocchio.visualize import MeshcatVisualizer          
import csv
import os        
import numpy as np
import pandas as pd
import time

class Vis:
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
        urdf_path = os.path.join(current_path, '../../phc/data/assets/robot/N2/urdf', 'N2.urdf')
        urdf_dirs = os.path.join(current_path, '../../phc/data/assets/robot/N2/urdf')
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path,
                                                    root_joint = self.joint_model,
                                                    package_dirs = urdf_dirs)
        
        script_directory = os.path.dirname(os.path.abspath(__file__))
        output_directory = os.path.join(script_directory, 'data')
        file_path = os.path.join(output_directory, 'output.csv')
        self.motions_path = file_path
        
        self.mixed_jointsToLockIDs =[]
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )
        
        self.vis = None

        if self.Visualization:
            self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
            self.vis.initViewer(open=True) 
            self.vis.loadViewerModel("pinocchio") 
            self.vis.displayFrames(True, frame_ids=[101, 102], axis_length = 0.15, axis_width = 5)
            self.vis.display(pin.neutral(self.reduced_robot.model))
    
    def vis_from_csv(self):
            csv_reader = pd.read_csv(self.motions_path)
            # 按行迭代数据
            time_last = 0
            for index, row in csv_reader.iterrows():
                # print(f"Row {index}: {row.tolist()}") 
                sol_q = row.tolist()[2:]
                time_now = row[1]
                if self.Visualization:
                    self.vis.display(np.array(sol_q))  # for visualization 
                time.sleep((time_now - time_last)*3)
                time_last = time_now


if __name__ == '__main__':
    visualiation = Vis(True)
    visualiation.vis_from_csv()
    