#!/usr/bin/env python3
from ur_controller import URrobot
from controller_ours_rbf import Controller  # change the controller script
# from controller_ours_GNN_NN import Controller  # change the controller script
# from controller_ours_GNN_NN_region import Controller  # change the controller script
from dlo_detector import dloDetector
from dlo_detector import load_config
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import pyrealsense2 as rs
import utils.print_utils as print_utils
import sys
import os
# import file_utils
import time
# import controller_viz_node
# import controller_prederror_node
# import rospy
from utils.state_index import I
from get_config import args, args_ur

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

VMWARE_URSIM_IP1 = '172.16.45.129'
VMWARE_URSIM_IP2 = '172.16.45.128'
NTU_UR5E_MAIN_IP = '10.149.230.1'
NTU_UR5E_LOAN_IP = '10.149.230.2'

# config_path = os.path.expanduser("~/data/Downloads/xinge/data_51/data_config.json")
robot_ip_left = NTU_UR5E_LOAN_IP # ip addr of left robot eff
robot_ip_right = NTU_UR5E_MAIN_IP # ip addr of right robot eff

dim = 2

PIXEL_TO_M_X = 0.0016 # 0.125mm
PIXEL_TO_M_Y = 0.0016 # 0.125mm

# load transformation matrix from config
# dlo_config = load_config(config_path)
# TRANSFORM_MATRIX = {"tf_img_rb1": dlo_config["tf_img_rb1"],
#                     "tf_rb2_rb1": dlo_config["tf_rb2_rb1"]}
# POSE_START1 = dlo_config["pose_start1"]
# POSE_START2 = dlo_config["pose_start2"]
#
# dlo_length = dlo_config["dlo_length"] # in m

TRANSFORM_MATRIX = {"tf_img_rb1": args_ur.tf_img_rb1_2D_onearm,
                    "tf_rb2_rb1": args_ur.tf_rb2_rb1_2D_onearm}
# POSE_START1 = args_ur.pose_start1_2D
POSE_START2 = args_ur.pose_start2_2D_onearm

dlo_length = args_ur.dlo_length # in m

project_dir = args.project_dir
target_idx = args.controller_object_fps_idx
env_dim = args.env_dimension

desired_pos_file = project_dir + "/env_dlo/env_" + env_dim + "_Data/StreamingAssets/desired_shape/" + env_dim  + "/" + "realsim_desired_positions_onearm.npy"
desired_pose = np.load(desired_pos_file).reshape(-1, 10, 3)
nr_of_cases = desired_pose.shape[0]

# motion profiles for moving eff
ACC = 0.1
VEL = 0.03
DURATION = 0.1 * 5
JOG_TIMEOUT = 30 # 30 seconds
CONTROL_INTERVAL = 0.1 #fix control interval for control

img_name = "realsense_camera"


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

    # left_eff_pose = left_eff.receive.getActualTCPPose() # change to desried pos
    left_eff_pose = desired_pose.flatten()[0:3]
    right_eff_pose = transformRobot(right_eff.receive.getActualTCPPose(), TRANSFORM_MATRIX["tf_rb2_rb1"], dim=dim) # transform to left robot base frame
    if dim == 2 : # Normalize Z
        left_eff_pose[2] = 0
        right_eff_pose[2] = 0
    # left_eff_quat = R.from_euler('xyz', left_eff_pose[3:]).as_quat()
    left_eff_quat = [0,0,0,0]
    # left_eff_vel = np.array(left_eff.receive.getActualTCPSpeed())
    left_eff_vel = [0,0,0,0,0,0]
    right_eff_quat = R.from_euler('xyz', right_eff_pose[3:]).as_quat()
    right_eff_vel = np.array(right_eff.receive.getActualTCPSpeed())

    state[0] = dlo_length
    state[1:4] = left_eff_pose[:3] # first feature point is left eff pos
    state[4:28] = keypoints_vector.flatten()
    state[28:31] = right_eff_pose[:3] # last feature point is right eff pos
    state[31:34] = left_eff_pose[:3]
    state[34:38] = left_eff_quat
    state[38:41] = right_eff_pose[:3]
    state[41:45] = right_eff_quat
    state[45:48] = left_eff_vel[:3] # first feature point vel is left eff vel
    state[48:72] = keypoints_vel.flatten()
    state[72:75] = right_eff_vel[:3] # last feature point vel is right eff vel
    state[75:81] = left_eff_vel
    state[81:87] = right_eff_vel
    state[87:117] = desired_pose.flatten()

    return state.astype(np.float32)


def drawRegionOnImage(img, desired_pose, region=None):
    if region is None:
        region = [0.0]*len(desired_pose)
    idx = 0
    for (x,y,z), r in zip(desired_pose, region):
        thickness = 1
        if idx in target_idx:
            r /= np.mean([PIXEL_TO_M_X, PIXEL_TO_M_Y]) # convert to pixel
            if r == 0:
                thickness = -1
                r = 3
            cv2.circle(img, (int(x), int(y)), radius=int(r), color=(0, 255, 0), thickness=thickness)
        idx += 1
    return img

def rsGetMetric(keypoints_vector, aligned_depth_frame, color_intrinsics, dim=2):
    keypoints_vector_in_m = []

    # Get color intrinsics
    for (x, y, z) in keypoints_vector:
        depth = aligned_depth_frame.get_distance(int(x), int(y))
        if dim == 3:
            if depth == 0:  # Check if the depth is valid
                print_utils.logwarn(f"Invalid depth at pixel ({x}, {y}), skipping this keypoint.")
                continue
        depth_point = rs.rs2_deproject_pixel_to_point(color_intrinsics, [x, y], depth)
        keypoints_vector_in_m.append([ depth_point[0], -depth_point[1], depth_point[2] ]) # negative y because y is flipped in image

    return np.array(keypoints_vector_in_m).flatten()

def pixelToMetric(pixels):
    if isinstance(pixels, tuple):
        pixels = [pixels]
    coords = []
    for (u,v) in pixels:
        x = u * PIXEL_TO_M_X
        y = v * PIXEL_TO_M_Y
        coords.append((x,y))
    return coords

def transformRobotToImage(coords, tf_matrix, dim=2):
    ret = []

    for x,y,z in coords:
        point = np.array([x, y, z, 1])  # homogeneous coordinate
        new_point = np.linalg.inv(tf_matrix) @ point
        x_new, y_new, z_new, _ = new_point
        ret.append((x_new, y_new, z_new))

    return np.array(ret)

def transformImageToRobot(pixel_coords, tf_matrix, dim=2):
    ret = []
    for x,y,z in pixel_coords:
        point = np.array([x, y, z, 1])  # homogeneous coordinate
        new_point = tf_matrix @ point
        x_new, y_new, z_new, _ = new_point
        ret.append((x_new, y_new, z_new))
    return np.array(ret)

def transformRobot(pose, tf_matrix, dim=2):
        x, y, z, tx, ty, tz = pose
        point = np.array([x, y, z, 1])  # homogeneous coordinate
        new_point = tf_matrix @ point
        x_new, y_new, z_new, _ = new_point
        return np.array([x_new, y_new, z_new, tx, ty, tz])

# def resetEffPose(robot1, robot2, img=None):
#     print_utils.logwarn(f"Moving robot1 to : {POSE_START1}.\nMoving robot2 to : {POSE_START2}. Please ensure no collision! Press Enter to continue")
#     if img is not None:
#         while True:
#             img_copy = img.copy()
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('\r'):
#                 break
#             cv2.putText(img_copy, f"Press enter to reset robot", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
#             cv2.imshow(img_name,img_copy)
#         img_copy = img.copy()
#         cv2.putText(img_copy, f"Robot resetting, please wait.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
#         cv2.imshow(img_name,img_copy)
#         cv2.waitKey(1)
#     else:
#         input()
#     robot1.pointToPointMove(POSE_START1, VEL, ACC, wait=False)
#     robot2.pointToPointMove(POSE_START2, VEL, ACC, wait=False)
#     robot1.waitStop()
#     robot2.waitStop()
#     success1 = robot1.isAtPose(POSE_START1)
#     success2 = robot2.isAtPose(POSE_START2)
#     if not success1:
#         print_utils.logwarn(f"Robot1 failed to reach start pose: {POSE_START1}")
#     if not success2:
#         print_utils.logwarn(f"Robot2 failed to reach start pose: {POSE_START2}")
#     return success1 & success2

def startVideoWriter(project_dir, env_dim, control_law, case_id):
    result_dir = os.path.join(project_dir, "results", "real", "control", control_law, env_dim) + "/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    video_path = result_dir + f"video_con_occ_2_{case_id}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 480))
    print_utils.loginfo(f"Recording video to: {video_path}")
    return writer


def main():
    pipeline_started = False
    first_instance = True
    frames_to_skip = 50
    keep_going = False
    velocity_mode = None
    case_id = 0
    video_writer = None

    try:
        # dlo = dloDetector(config_path)
        dlo = dloDetector("C:\\Users\\User\pythonProject\DLO_control\ws_dlo\src\dlo_manipulation_pkg\scripts\\utils\config_tf.json")
        # left_eff = URrobot(robot_ip_left)
        right_eff = URrobot(robot_ip_right)
        controller = Controller()

        # controller.initializePublisher() # initiate the dlo publisher
        # pub = controller_viz_node.initializePublisher()  # Initialize the publisher for visualization
        # controller_viz_node.publishResetPlot(pub, timeout=5)  # Reset the plot before starting
        # pub2 = controller_prederror_node.initializePublisher()  # Initialize the publisher for prediction error
        # controller_prederror_node.publishResetPlot(pub2, timeout=5)  # Reset the plot before starting

        keypoints_matrix = []
        # region = controller.region

        # if not (left_eff.is_online and right_eff.is_online):
        if not (right_eff.is_online):
            print_utils.logerr('Either Robot is not online!')
            return

        # resetEffPose(left_eff, right_eff)
        right_eff.pointToPointMove(POSE_START2, VEL, ACC)
        # start the pipeline
        print_utils.loginfo("Starting realsense camera...")
        pipeline.start(config)
        # Create alignment primitive with color as its target stream:
        align = rs.align(rs.stream.color)
        print_utils.loginfo("camera started!")
        pipeline_started = True

        time_start = time.time()
        send_control_starttime = CONTROL_INTERVAL  # just to initialize first instance of control starttime
        while True:
            print_utils.loginfo(f"Program Interval: {int((time.time() - time_start) * 1000)}ms")
            time_start = time.time()
            # print_utils.loginfo(f"iter: {idx}")
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
            img_raw = img.copy()

            if frames_to_skip:  # skip the first few frames
                # print_utils.loginfo(f"Skipping unstable frames, {frames_to_skip} left...")
                frames_to_skip -= 1
                cv2.putText(img, f"Loading... please wait.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.imshow(img_name, img)
                cv2.waitKey(1)
                continue

            labels = np.array(target_idx)[~np.isin(target_idx, [0, 9])]  # remove eff labels
            img_with_fp, keypoints_vector, keypoints_matrix,missing_idx = dlo.detectFeatures(img, keypoints_matrix, dim=2, labels=labels,
                                                                                 nolabel=False)
            keypoints_vector_in_m = transformImageToRobot(keypoints_vector, TRANSFORM_MATRIX["tf_img_rb1"],
                                                          dim=dim)  # transform to robot base frame
            # keypoints_vector_in_m = rsGetMetric(keypoints_vector, aligned_depth_frame, color_intrinsics)
            # keypoints_vector *= PIXEL_TO_M  # convert to m
            print_utils.loginfo(f"\nkeypoints in m: {keypoints_vector_in_m}")
            img_fp_and_region = drawRegionOnImage(img_with_fp, transformRobotToImage(desired_pose[case_id],
                                                                                     TRANSFORM_MATRIX["tf_img_rb1"],
                                                                                     dim=dim), region=None) #controller.region
            # cv2.imshow(img_paths[idx],img_fp_and_region)
            img_fp_and_region = dlo.showDetectionRegion(img_fp_and_region)
            cv2.putText(img_fp_and_region, f"case {case_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.imshow(img_name, img_fp_and_region)
            if video_writer is not None:           # ← ADD THIS
                video_writer.write(img_fp_and_region) 

            if first_instance:
                print_utils.loginfo("First instance, initializing camera matrix...")
                prev_keypoints_vector = keypoints_vector_in_m
                first_instance = False
                if not np.any(keypoints_vector):  # if no keypoints detected during first instance, retry
                    print_utils.logwarn("No keypoints detected, retrying...")
                    keypoints_matrix = []
                    if cv2.waitKey(1) == ord('q'):  # pause until any key is pressed
                        print_utils.loginfo("Exiting program...")
                        break
                    first_instance = True
                continue

            # desired_pose_in_m = transformImageToRobot(desired_pose[case_id], TRANSFORM_MATRIX["tf_img_rb1"], dim=dim) # transform to robot base frame
            # desired_pose_in_m = rsGetMetric(desired_pose_transformed, aligned_depth_frame, color_intrinsics)

            state = getState(dlo_length,
                             None, right_eff,
                             keypoints_vector_in_m, prev_keypoints_vector,
                             # desired_pose_in_m,
                             desired_pose[case_id],
                             dim=dim)


            prev_keypoints_vector = keypoints_vector_in_m
            state[
                I.left_end_avel_idx + I.right_end_avel_idx] /= 2 * np.pi  # change unit of the output angular velocity from rad/s to rev/s
            # print_utils.loginfo(f"State:\n {state}")

            #Change here
            # control_input = controller.generateControlInput(state, state, missing_idx=missing_idx)
            # control_input = controller.generateControlInput(state, state)
            control_input = controller.generateControlInput(state)
            print(control_input)
            control_input[[3, 4, 5, 9, 10,
                           11]] *= 2 * np.pi  # change the unit of the output angular velocity from 2pi*rad/s(rev/s) to rad/s

            print(f"Robot Control Input: {control_input}")

            print(
                "'ENTER' to start move\n'SPACE' to pause \n'x' to move one step\n'q' to quit, \n'r' to refresh keypoints, \n'n' to next iteration\n\n")

            if keep_going:
                key = cv2.waitKey(1)
                if time.time() - jog_starttime > JOG_TIMEOUT:
                    print_utils.loginfo(f"Jog Stopped, timeout: {JOG_TIMEOUT:.3f}")
                    keep_going = False
                    if velocity_mode == 'jog':
                        velocity_mode = None
                        # left_eff.velocityJogControlStop()
                        right_eff.velocityJogControlStop()
            else:
                keep_going = False
                key = cv2.waitKey(0)
                # key = -1
                # while key == -1:
                #         # keep refreshing the display while waiting for keypress
                #         frame = pipeline.wait_for_frames()
                #         aligned_frames = align.process(frame)
                #         color_frame = aligned_frames.get_color_frame()
                #         if color_frame:
                #             img = np.asanyarray(color_frame.get_data())
                #             img_with_fp, keypoints_vector, keypoints_matrix, missing_idx = dlo.detectFeatures(img, keypoints_matrix)
                #             cv2.imshow(img_name, img_with_fp)
                #         key = cv2.waitKey(30) & 0xFF

            if key == ord('\r') and keep_going == False:
                print_utils.logwarn("Robot will commence motion, please ensure the workspace is clear! Press enter again to confirm...")
                while cv2.waitKey(0) != ord('\r'):
                    pass
                print_utils.logwarn("Robot moving, keep clear of the workspace!")
                time.sleep(2)
                velocity_mode = 'jog' # 'step' or 'jog'
                keep_going = True
                jog_starttime = time.time()
                if video_writer is None:                                                              # ← ADD
                    video_writer = startVideoWriter(project_dir, env_dim, args.controller_control_law, case_id)            # ← ADD

            elif key == ord('x'):
                print_utils.loginfo("Moving one step...")
                keep_going = False
                velocity_mode = 'step' # 'step' or 'jog'
            elif key == ord(' '):
                print_utils.loginfo("Pausing robot motion...")
                keep_going = False
                if velocity_mode == 'jog':
                    velocity_mode = None
                    # left_eff.velocityJogControlStop()
                    right_eff.velocityJogControlStop()
            elif key == ord('q'): # pause until any key is pressed
                print_utils.loginfo("Exiting program...")
                if velocity_mode == 'jog':
                    velocity_mode = None
                    # left_eff.velocityJogControlStop()
                    right_eff.velocityJogControlStop()
                if video_writer is not None:                                         # ← ADD
                    video_writer.release()                                           # ← ADD
                    video_writer = None                                              # ← ADD
                    print_utils.loginfo(f"Video saved for case {case_id}")
                controller.reset(state)          # ← ADD
                break
            elif key == ord('r'): # refresh
                print_utils.loginfo("Refreshing keypoints...")
                keep_going = False
                keypoints_matrix = []
                first_instance = True
                #controller.reset(state)
                if velocity_mode == 'jog':
                    velocity_mode = None
                    # left_eff.velocityJogControlStop()
                    right_eff.velocityJogControlStop()
            elif key == ord('n'): # next iteration
                if video_writer is not None:                                         # ← ADD
                    video_writer.release()                                           # ← ADD
                    video_writer = None                                              # ← ADD
                    print_utils.loginfo(f"Video saved for case {case_id}")
                case_id += 1
                if case_id >= nr_of_cases:
                    print_utils.loginfo("All cases completed!")
                    break
                print_utils.loginfo("Next iteration")
                keep_going = False
                keypoints_matrix = []
                first_instance = True
                controller.reset(state)
                if velocity_mode == 'jog':
                    velocity_mode = None
                    # left_eff.velocityJogControlStop()
                    right_eff.velocityJogControlStop()
                # resetEffPose(left_eff, right_eff, img_raw)
                right_eff.pointToPointMove(POSE_START2, VEL, ACC)

            # Move robot for 0.1s towards goal
            if velocity_mode == 'step':
                # left_eff.velocityControl(control_input[:6], ACC, DURATION)
                right_eff.velocityControl(control_input[6:], ACC, DURATION)
            elif velocity_mode == 'jog':
                while (time.time() - send_control_starttime < CONTROL_INTERVAL):
                    continue
                print_utils.loginfo(f"Control Interval: {int( ( time.time() - send_control_starttime )*1000)}ms")
                # left_eff.velocityJogControl(control_input[:6], ACC, timeout=2)
                right_eff.velocityJogControl(control_input[6:], ACC, timeout=2)
                send_control_starttime = time.time()


    except Exception as e:
        tb = sys.exc_info()[2]  # traceback object
        print_utils.logerr(f"LINE {tb.tb_lineno}: {e}")

    finally:
        # Stop the pipeline
        if pipeline_started:
            pipeline.stop()
        if video_writer is not None:   # ← ADD
            video_writer.release()   
        # left_eff.terminate()
        right_eff.terminate()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # rospy.init_node("ur_dlomanipulation_node")
    # rospy.loginfo("UR DLO Manipulation Node initialized successfully.")
    print_utils.loginfo('Start control script')
    main()
    print_utils.loginfo('End of script')

