import pandas as pd
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.interpolate import interp1d
import os
import json
joint_names = [
    'l_arm_shoulder_pitch_joint','l_arm_shoulder_roll_joint','l_arm_shoulder_yaw_joint','l_arm_elbow_joint','l_leg_hip_yaw_joint',\
    'l_leg_hip_roll_joint','l_leg_hip_pitch_joint','l_leg_knee_joint','l_leg_ankle_joint','r_arm_shoulder_pitch_joint','r_arm_shoulder_roll_joint',\
    'r_arm_shoulder_yaw_joint','r_arm_elbow_joint','r_leg_hip_yaw_joint','r_leg_hip_roll_joint','r_leg_hip_pitch_joint','r_leg_knee_joint','r_leg_ankle_joint'
]

TARGET_MOTION_FILE = "/home/djh/PHC/scripts/data_process/data/run_motion.json"

# 获取当前脚本的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 构建CSV文件的路径
csv_path = os.path.join(script_dir, '.', 'data', 'output.csv')
# 读取CSV文件
df = pd.read_csv(csv_path)
# 假设欧拉角在CSV中的列名为 'roll', 'pitch', 'yaw'
euler_angles = df[['roll', 'pitch', 'yaw']].values
rotation = R.from_euler('xyz', euler_angles, degrees=False)
quaternions = rotation.as_quat()  # 转换为四元数 (xyzw)




# 计算线速度
positions = df[['posX','posY','posZ']].values
dt = 1 / 60  # 60Hz的帧间隔
linear_velocity = np.gradient(positions, axis=0) / dt


euler_derivatives = np.diff(euler_angles, axis=0) / dt
euler_derivatives = np.vstack((np.zeros((1, 3)), euler_derivatives))  # 第一帧补零

# 初始化角速度数组
angular_velocity_world = np.zeros((len(euler_angles), 3))

# 遍历每一帧，计算角速度
for i in range(len(euler_angles)):
    phi, theta, psi = euler_angles[i]  # 当前帧的欧拉角
    dphi, dtheta, dpsi = euler_derivatives[i]  # 当前帧的欧拉角导数

    # 构建旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi), np.cos(phi)]])

    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])

    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi), np.cos(psi), 0],
                   [0, 0, 1]])

    # 组合旋转矩阵
    R = Rz @ Ry @ Rx

    # 计算角速度
    angular_velocity_body = np.array([dphi, dtheta, dpsi])
    angular_velocity_world[i] = R @ angular_velocity_body

# 插值到 200Hz
# 原始时间轴 (60Hz)
t_60hz = np.arange(len(euler_angles)) / 60.0

# 新的时间轴 (200Hz)
t_200hz = np.arange(0, t_60hz[-1], 1 / 200.0)

# 插值角速度



# print(quaternions[0,:])

# # 计算角速度
# quat_diff = np.diff(quaternions, axis=0)
# quat_diff = np.vstack((np.array([0,0,0,1]), quat_diff))  # 在第一帧补零
# dq_dt = quat_diff / dt  # 四元数对时间的导数

# angular_velocity = np.zeros((len(quaternions), 3))  # 初始化角速度数组

# for i in range(len(quaternions)):
#     q = quaternions[i]  # 当前四元数
#     dq = dq_dt[i]  # 当前四元数的导数

#     # 计算四元数的逆
#     q_conj = np.array([-q[0], -q[1], -q[2], q[3]])  # 四元数的共轭（逆）
    
#     # 计算角速度: ω = 2 * dq * q^{-1}
#     omega = 2 * R.from_quat(dq).as_rotvec()  # 使用 scipy 的 Rotation 类计算角速度
#     angular_velocity[i] = omega


# 原始时间轴 (60Hz)
t_60hz = np.arange(len(df)) / 60.0

# 新的时间轴 (200Hz)
t_200hz = np.arange(0, t_60hz[-1], 1/200.0)

# 插值函数
interp_positions = interp1d(t_60hz, positions, axis=0, kind='linear')
interp_quaternions = interp1d(t_60hz, quaternions, axis=0, kind='linear')
interp_linear_velocity = interp1d(t_60hz, linear_velocity, axis=0, kind='linear')
interp_angular_velocity = interp1d(t_60hz, angular_velocity_world, axis=0, kind='linear')


# 插值后的数据
positions_200hz = interp_positions(t_200hz)
quaternions_200hz = interp_quaternions(t_200hz)
linear_velocity_200hz = interp_linear_velocity(t_200hz)
angular_velocity_200hz = interp_angular_velocity(t_200hz)
# 计算关节速度
joint_positions = df[joint_names].values
joint_velocity = np.gradient(joint_positions, axis=0) / dt

# 插值关节位置和速度
interp_joint_positions = interp1d(t_60hz, joint_positions, axis=0, kind='linear')
interp_joint_velocity = interp1d(t_60hz, joint_velocity, axis=0, kind='linear')

joint_positions_200hz = interp_joint_positions(t_200hz)
joint_velocity_200hz = interp_joint_velocity(t_200hz)


frames = np.hstack((positions_200hz,quaternions_200hz,linear_velocity_200hz,angular_velocity_200hz,joint_positions_200hz,joint_velocity_200hz)).tolist()


outputDict = {
            "LoopMode": "none",  # "none" or "wrap"
            "FrameDuration": 1/200,
            "EnableCycleOffsetPosition":"true",
            "EnableCycleOffsetRotation":"true",
            "MotionWeight": 1.0,
            "Frames": frames
}


with open(TARGET_MOTION_FILE, "w") as output:
    output.write(json.dumps(outputDict, indent=4))
    
print("Generating " + TARGET_MOTION_FILE)