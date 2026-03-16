#!/usr/bin/env python3

# Created Jan 6 2026. Remi. To generate desired shape for dlo manipulation 2D

from ur_controller import URrobot
from controller_ours_v1 import Controller
from dlo_detector import dloDetector
# from dlo_detector import load_config
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import pyrealsense2 as rs
import utils.print_utils as print_utils
import sys
import os
# import utils.file_utils as file_utils   
import time
# import controller_viz_node
# import rospy
from utils.state_index import I
from get_config import args, args_ur

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

NTU_UR5E_MAIN_IP = '10.149.230.1'
NTU_UR5E_LOAN_IP = '10.149.230.2'

# parameters to move the robot arms 
SPEED = 0.05                # m/s
R_SPEED = 2.222 / 5        # rad/s
R_ACC = 2.222 / 5          # rad/2
ACC = 0.1                 # acceleration
DURATION = 0.2              # how long each jog runs (s)

PIXEL_TO_M_X = 0.0016 # 0.125mm
PIXEL_TO_M_Y = 0.0016 # 0.125mm

img_name = "realsense_camera"

# config_path = os.path.expanduser("~/data/Downloads/xinge/data_51/data_config.json")
robot_ip_left = NTU_UR5E_LOAN_IP # ip addr of left robot eff
robot_ip_right = NTU_UR5E_MAIN_IP # ip addr of right robot eff

desired_pose = np.array([[(167.0, 167.0, 0.0),
                        (186.0, 139.0, 0.0),
                        (192.0, 106.0, 0.0),
                        (193.0, 73.0, 0.0),
                        (205.0, 44.0, 0.0),
                        (238.0, 40.0, 0.0),
                        (266.0, 66.0, 0.0),
                        (278.0, 93.0, 0.0),
                        (292.0, 119.0, 0.0),
                        (325.0, 118.0, 0.0)]])

# config_path = os.path.expanduser("~/data/Downloads/xinge/data_51/data_config.json")
config_path = os.path.expanduser("C:/Users/91990/Documents/GitHub/FYP_Object_Detection_Model/shape_control_DLO_2/ws_dlo/src/dlo_system_pkg/config/config.json")
dim = 2
dlo_length = 0.5 # in m

# load transformation matrix from config
# dlo_config = load_config(config_path)

# if dim == 2:
#     TRANSFORM_MATRIX = {"tf_img_rb1": dlo_config["tf_img_rb1_2D"],
#                         "tf_rb2_rb1": dlo_config["tf_rb2_rb1_2D"]}
#     POSE_START1 = dlo_config["pose_start1_2D"]
#     POSE_START2 = dlo_config["pose_start2_2D"]
# elif dim == 3:
#     TRANSFORM_MATRIX = {"tf_img_rb1": dlo_config["tf_img_rb1_3D"],
#                         "tf_rb2_rb1": dlo_config["tf_rb2_rb1_3D"]}
#     POSE_START1 = dlo_config["pose_start1_2D"]
#     POSE_START2 = dlo_config["pose_start2_2D"]


TRANSFORM_MATRIX = {"tf_img_rb1": args_ur.tf_img_rb1_2D_onearm,
                    "tf_rb2_rb1": args_ur.tf_rb2_rb1_2D_onearm}
POSE_START1 = args_ur.pose_start1_2D
POSE_START2 = args_ur.pose_start2_2D_onearm

project_dir = args.project_dir
target_idx = args.controller_object_fps_idx
env_dim = args.env_dimension

def getState(dlo_length, left_eff, right_eff, keypoints_vector, prev_keypoints_vector, desired_pose, dim=2):
    state = np.zeros(117)
    """
        The state is a 117-dimension vector.

        0: the length of the DLO
        1~30: the positions of the 10 features (10*3)
        31~44: the pose of the two end-effectors
        left end positions (3) + left end orientation (4) + right end position (3) + right end orientation (4)
        The representation of the orientations is quaternion.
        45~74: the velocities of the 10 features (10*3)
        75~86: the velocities of the end-effectors
        left end linear velocity (3) + left end angular velocity (3) + right end linear velocity (3) + right end angular velocity (3)
        The representation of the angular velocities is rotation vector.
        87~116: the desired positions of the 10 features (10*3)

        Note that in 2D environment, the dimension of the position of one feature is still three, but the value in the z axis is always zero. 
    """
    keypoints_vel = keypoints_vector - prev_keypoints_vector

    # left_eff_pose = left_eff.receive.getActualTCPPose()
    right_eff_pose = transformRobot(right_eff.receive.getActualTCPPose(), TRANSFORM_MATRIX["tf_rb2_rb1"], dim=dim) # transform to left robot base frame
    if dim == 2 : # Normalize Z
        # left_eff_pose[2] = 0
        right_eff_pose[2] = 0
    # left_eff_quat = R.from_euler('xyz', left_eff_pose[3:]).as_quat()
    # left_eff_vel = left_eff.receive.getActualTCPSpeed()
    right_eff_quat = R.from_euler('xyz', right_eff_pose[3:]).as_quat()
    right_eff_vel = right_eff.receive.getActualTCPSpeed()

    # Left end is physically fixed — constant pose in right robot base frame
    left_eff_pose = [-0.6629, 0.5021, 0.050, 2.084, 2.418, 0.0]
    left_eff_vel  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    left_eff_quat = R.from_euler('xyz', left_eff_pose[3:]).as_quat()


    state[0] = dlo_length
    # state[1:4] = left_eff_pose[:3] # first feature point is left eff pos
    state[1:4] = left_eff_pose[:3] # first feature point is left eff pos
    state[4:28] = keypoints_vector.flatten()
    state[28:31] = right_eff_pose[:3] # last feature point is right eff pos
    state[31:34] = left_eff_pose[:3]
    # state[34:38] = left_eff_quat
    state[34:38] = [0,0,0,0]
    state[38:41] = right_eff_pose[:3]
    state[41:45] = right_eff_quat
    # state[45:48] = left_eff_vel[:3] # first feature point vel is left eff vel
    state[45:48] = [0,0,0] # first feature point vel is left eff vel
    state[48:72] = keypoints_vel.flatten()
    state[72:75] = right_eff_vel[:3] # last feature point vel is right eff vel
    # state[75:81] = left_eff_vel
    state[75:81] = [0,0,0,0,0,0]
    state[81:87] = right_eff_vel
    state[87:117] = desired_pose.flatten()

    return state.astype(np.float32)

def drawRegionOnImage(img, desired_pose, region=None):
    thickness = 1
    if region is None:
        region = [10]*len(desired_pose) # default radius 10
        thickness = -1 # filled circle
    idx = 0
    for (x,y,z), r in zip(desired_pose, region):
        idx += 1
        if idx == 1: # skip first and last point (robot end effectors)
            continue
        elif idx == len(desired_pose):
            continue
        r /= np.mean([PIXEL_TO_M_X, PIXEL_TO_M_Y]) # convert to pixel
        cv2.circle(img, (int(x), int(y)), radius=int(r), color=(0, 255, 0), thickness=thickness)
    return img

def transformImageToRobot(pixel_coords, tf_matrix, dim=2):
    ret = []
    for x,y,z in pixel_coords:
        point = np.array([x, y, z, 1])  # homogeneous coordinate
        new_point = tf_matrix @ point
        x_new, y_new, z_new, _ = new_point
        ret.append((x_new, y_new, z_new))
    return np.array(ret)

def transformRobotToImage(pose, tf_matrix):
    tf_inv = np.linalg.inv(tf_matrix)
    x, y, z, tx, ty, tz = pose
    point = np.array([x, y, z, 1])  # homogeneous
    new_point = tf_inv @ point
    x_new, y_new, z_new, _ = new_point
    return np.array([x_new, y_new, z_new, tx, ty, tz])

def transformRobot(pose, tf_matrix, dim=2):
        x, y, z, tx, ty, tz = pose
        point = np.array([x, y, z, 1])  # homogeneous coordinate
        new_point = tf_matrix @ point
        x_new, y_new, z_new, _ = new_point
        return np.array([x_new, y_new, z_new, tx, ty, tz])

def main(): 
    pipeline_started = False
    first_instance = True
    frames_to_skip = 10
    keep_going = False
    velocity_mode = None
    

    try:
        dlo = dloDetector(config_path)
        # left_eff = URrobot(robot_ip_left)
        right_eff = URrobot(robot_ip_right)
        # robot = left_eff
        controller = Controller()
        keypoints_matrix = []

        if not (right_eff.is_online):
            print_utils.logerr('Either Robot is not online!')
            return
        
        # save_dir = controller.project_dir + "/env_dlo/env_" + controller.env_dim + "_Data/StreamingAssets/desired_shape/" + controller.env_dim  + "/"
        save_dir = project_dir + "/env_dlo/env_" + env_dim + "_Data/StreamingAssets/desired_shape/" + env_dim + "/"
        img_dir = save_dir + "images/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        savefile = save_dir + "realsim_desired_positions_onearm.npy"
        if os.path.exists(savefile):
            data = np.load(savefile)
            case_id = data.shape[0] 
        else:
            case_id = 0

        # start the pipeline
        print_utils.loginfo("Starting realsense camera...")
        pipeline.start(config)
        # Create alignment primitive with color as its target stream:
        align = rs.align(rs.stream.color)
        print_utils.loginfo("camera started!")
        pipeline_started = True

        while True:
            time_start = time.time()
            #print_utils.loginfo(f"iter: {idx}")
            frame = pipeline.wait_for_frames()
            aligned_frames = align.process(frame)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            if not aligned_depth_frame or not color_frame:
                print_utils.logwarn("No frame received from realsense")
                continue
            # Convert to numpy array (OpenCV format)
            img = np.asanyarray(color_frame.get_data())     

            
            if frames_to_skip: # skip the first few frames
                frames_to_skip -= 1
                cv2.putText(img, f"Loading... please wait.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.imshow(img_name,img)
                cv2.waitKey(1)
                continue

            img_with_fp, keypoints_vector, keypoints_matrix, missing_idx = dlo.detectFeatures(img, keypoints_matrix)
            keypoints_vector_in_m = transformImageToRobot(keypoints_vector, TRANSFORM_MATRIX["tf_img_rb1"], dim=dim) # transform to robot base frame
            #keypoints_vector_in_m = rsGetMetric(keypoints_vector, aligned_depth_frame, color_intrinsics)
            #keypoints_vector *= PIXEL_TO_M  # convert to m
            print_utils.loginfo(f"\nkeypoints in m: {keypoints_vector_in_m}")
            #img_fp_and_region = drawRegionOnImage(img_with_fp, desired_pose[case_id], region)
            #cv2.imshow(img_paths[idx],img_fp_and_region)
            img_fp_and_region = dlo.showDetectionRegion(img_with_fp)
            cv2.putText(img_fp_and_region, f"case {case_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.imshow(img_name,img_fp_and_region)

            if first_instance:
                print_utils.loginfo("First instance, initializing camera matrix...")
                prev_keypoints_vector = keypoints_vector_in_m
                first_instance = False
                if not np.any(keypoints_vector): # if no keypoints detected during first instance, retry
                    print_utils.logwarn("No keypoints detected, retrying...")
                    keypoints_matrix = []
                    if cv2.waitKey(1) == ord('q'): # pause until any key is pressed
                        print_utils.loginfo("Exiting program...")
                        break
                    first_instance = True
                continue

            desired_pose_in_m = transformImageToRobot(desired_pose[0], TRANSFORM_MATRIX["tf_img_rb1"], dim=dim) # transform to robot base frame
            #desired_pose_in_m = rsGetMetric(desired_pose_transformed, aligned_depth_frame, color_intrinsics)
            
            state = getState(dlo_length, 
                             None, right_eff,
                             keypoints_vector_in_m, prev_keypoints_vector, 
                             desired_pose_in_m,
                             dim=dim)
            prev_keypoints_vector = keypoints_vector_in_m
            
            fps_positions = state[I.fps_pos_idx]
            print_utils.loginfo(f"\nfps positions: \n {fps_positions}\n")
            
            print("\n\n'ENTER' to save goal \n'r' to reset keypoints \n'w a s d' to translate, 'n m' to rotate robot '1 or 2'\n 'q' to quit\n\n")

            # key = cv2.waitKey()
            #CHANGE HERE 
            key = cv2.waitKey(30) & 0xFF
            
            
            if key == ord('\r'): # save goal
                print_utils.loginfo(f"{case_id} saving data to: {savefile}") 
                if os.path.exists(savefile):
                    data = np.load(savefile)
                    data = np.vstack([data, fps_positions])
                    np.save(savefile, data)
                else: 
                    np.save(savefile, fps_positions)
                
                imgfile = img_dir + f"realsim_desired_positions_{case_id}.png"
                print_utils.loginfo(f"{case_id} saving image to: {imgfile}")
                cv2.imwrite(imgfile, img_fp_and_region)
                
                case_id += 1
                time.sleep(2)
                
            # elif key == ord('1'):
            #     robot = left_eff
            #     print_utils.loginfo("Controlling left eff")

            elif key == ord('2'):
                robot = right_eff
                print_utils.loginfo("Controlling right eff")

            elif key == ord('w'): # up
                robot.velocityControl([0, SPEED, 0, 0, 0, 0], ACC, DURATION)

            elif key == ord('s'): # down
                robot.velocityControl([0, -SPEED, 0, 0, 0, 0], ACC, DURATION)

            elif key == ord('a'): # left
                robot.velocityControl([-SPEED, 0, 0, 0, 0, 0], ACC, DURATION)

            elif key == ord('d'): # right
                robot.velocityControl([SPEED, 0, 0, 0, 0, 0], ACC, DURATION)

            elif key == ord('n'): # rotate
                robot.velocityControl([0, 0, 0, 0, 0, R_SPEED], R_ACC, DURATION)

            elif key == ord('m'):
                robot.velocityControl([0, 0, 0, 0, 0, -R_SPEED], R_ACC, DURATION)

            elif key == ord('q'): # quit
                print_utils.loginfo("Exiting program...")
                break
            
            elif key == ord('r'): # reset
                print_utils.loginfo("Resetting keypoints...")
                keep_going = False
                keypoints_matrix = []
                first_instance = True
                
            

    except Exception as e:
        tb = sys.exc_info()[2]  # traceback object
        print_utils.logerr(f"LINE {tb.tb_lineno}: {e}")
    
    finally:
        # Stop the pipeline
        if pipeline_started:
            pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # rospy.init_node("ur_dlomanipulation_node")
    # rospy.loginfo("UR DLO Manipulation Node initialized successfully.")
    #
    print_utils.loginfo('Start control script')
    main()
    print_utils.loginfo('End of script')
