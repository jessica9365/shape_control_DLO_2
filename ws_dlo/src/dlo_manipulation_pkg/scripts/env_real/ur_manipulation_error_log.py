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
import time
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


robot_ip_left = NTU_UR5E_LOAN_IP
robot_ip_right = NTU_UR5E_MAIN_IP

dim = 2

PIXEL_TO_M_X = 0.0016
PIXEL_TO_M_Y = 0.0016


TRANSFORM_MATRIX = {
    "tf_img_rb1": args_ur.tf_img_rb1_2D_onearm,
    "tf_rb2_rb1": args_ur.tf_rb2_rb1_2D_onearm
}
POSE_START2 = args_ur.pose_start2_2D_onearm

dlo_length = args_ur.dlo_length

project_dir = args.project_dir
target_idx = args.controller_object_fps_idx
env_dim = args.env_dimension

desired_pos_file = (
    project_dir
    + "/env_dlo/env_" + env_dim + "_Data/StreamingAssets/desired_shape/"
    + env_dim + "/" + "realsim_desired_positions_onearm.npy"
)
desired_pose = np.load(desired_pos_file).reshape(-1, 10, 3)
nr_of_cases = desired_pose.shape[0]


ACC = 0.1
VEL = 0.03
DURATION = 0.1 * 5
JOG_TIMEOUT = 30
CONTROL_INTERVAL = 0.1

img_name = "realsense_camera"


def getState(dlo_length, left_eff, right_eff, keypoints_vector, prev_keypoints_vector, desired_pose, dim=2):
    state = np.zeros(117)
    """
    The state is a 117-dimension vector.

    0: the length of the DLO
    1~30: the positions of the 10 features (10*3)
    31~44: the pose of the two end-effectors
    45~74: the velocities of the 10 features (10*3)
    75~86: the velocities of the end-effectors
    87~116: the desired positions of the 10 features (10*3)

    In 2D, z is always zero.
    """
    keypoints_vel = keypoints_vector - prev_keypoints_vector

    left_eff_pose = desired_pose.flatten()[0:3]
    right_eff_pose = transformRobot(
        right_eff.receive.getActualTCPPose(),
        TRANSFORM_MATRIX["tf_rb2_rb1"],
        dim=dim
    )

    if dim == 2:
        left_eff_pose[2] = 0
        right_eff_pose[2] = 0

    left_eff_quat = [0, 0, 0, 0]
    left_eff_vel = [0, 0, 0, 0, 0, 0]
    right_eff_quat = R.from_euler('xyz', right_eff_pose[3:]).as_quat()
    right_eff_vel = np.array(right_eff.receive.getActualTCPSpeed())

    state[0] = dlo_length
    state[1:4] = left_eff_pose[:3]
    state[4:28] = keypoints_vector.flatten()
    state[28:31] = right_eff_pose[:3]
    state[31:34] = left_eff_pose[:3]
    state[34:38] = left_eff_quat
    state[38:41] = right_eff_pose[:3]
    state[41:45] = right_eff_quat
    state[45:48] = left_eff_vel[:3]
    state[48:72] = keypoints_vel.flatten()
    state[72:75] = right_eff_vel[:3]
    state[75:81] = left_eff_vel
    state[81:87] = right_eff_vel
    state[87:117] = desired_pose.flatten()

    return state.astype(np.float32)


def getCurrentFeaturePoints(right_eff, keypoints_vector_in_m, desired_pose_case, dim=2):
    """
    Returns current feature points as shape (10, 3), ordered to match desired_pose[case_id].

    Assumptions:
      - feature 0 = left endpoint, approximated from desired pose in one-arm setup
      - features 1..8 = detected internal keypoints
      - feature 9 = right endpoint from robot TCP
    """
    current_points = np.zeros((10, 3), dtype=np.float32)

    left_eff_pose = desired_pose_case[0].copy()
    right_eff_pose = transformRobot(
        right_eff.receive.getActualTCPPose(),
        TRANSFORM_MATRIX["tf_rb2_rb1"],
        dim=dim
    )

    if dim == 2:
        left_eff_pose[2] = 0
        right_eff_pose[2] = 0

    kp = keypoints_vector_in_m.reshape(-1, 3)
    if kp.shape[0] != 8:
        raise ValueError(f"Expected 8 internal keypoints, got {kp.shape[0]}")

    current_points[0] = left_eff_pose[:3]
    current_points[1:9] = kp
    current_points[9] = right_eff_pose[:3]

    return current_points


def log_feature_metrics(robot, t, model_name, case_id, desired_points, current_points,
                        control_input, missing_idx):
    """
    desired_points: (10, 3)
    current_points: (10, 3)
    """
    err = desired_points - current_points
    err_norms = np.linalg.norm(err, axis=1)

    rmse_all = float(np.sqrt(np.mean(err ** 2)))
    mean_err = float(np.mean(err_norms))
    max_err = float(np.max(err_norms))
    endpoint_mean_err = float(np.mean(err_norms[[0, 9]]))
    inner_mean_err = float(np.mean(err_norms[1:9]))

    tcp_pose = transformRobot(
        robot.receive.getActualTCPPose(),
        TRANSFORM_MATRIX["tf_rb2_rb1"],
        dim=dim
    )

    row = {
        "t": float(t),
        "model": str(model_name),
        "case": int(case_id),
        "rmse_all": rmse_all,
        "mean_err": mean_err,
        "max_err": max_err,
        "endpoint_mean_err": endpoint_mean_err,
        "inner_mean_err": inner_mean_err,
        "missing_count": int(len(missing_idx) if missing_idx is not None else 0),
        "tcp_x": float(tcp_pose[0]),
        "tcp_y": float(tcp_pose[1]),
        "tcp_z": float(tcp_pose[2]),
    }

    for i in range(10):
        row[f"des_{i}_x"] = float(desired_points[i, 0])
        row[f"des_{i}_y"] = float(desired_points[i, 1])
        row[f"des_{i}_z"] = float(desired_points[i, 2])

        row[f"cur_{i}_x"] = float(current_points[i, 0])
        row[f"cur_{i}_y"] = float(current_points[i, 1])
        row[f"cur_{i}_z"] = float(current_points[i, 2])

        row[f"err_{i}_x"] = float(err[i, 0])
        row[f"err_{i}_y"] = float(err[i, 1])
        row[f"err_{i}_z"] = float(err[i, 2])
        row[f"err_{i}_norm"] = float(err_norms[i])

    for j in range(len(control_input)):
        row[f"u_{j}"] = float(control_input[j])

    robot.logs.append(row)


def drawRegionOnImage(img, desired_pose, region=None):
    if region is None:
        region = [0.0] * len(desired_pose)
    idx = 0
    for (x, y, z), r in zip(desired_pose, region):
        thickness = 1
        if idx in target_idx:
            r /= np.mean([PIXEL_TO_M_X, PIXEL_TO_M_Y])
            if r == 0:
                thickness = -1
                r = 3
            cv2.circle(img, (int(x), int(y)), radius=int(r), color=(0, 255, 0), thickness=thickness)
        idx += 1
    return img


def rsGetMetric(keypoints_vector, aligned_depth_frame, color_intrinsics, dim=2):
    keypoints_vector_in_m = []

    for (x, y, z) in keypoints_vector:
        depth = aligned_depth_frame.get_distance(int(x), int(y))
        if dim == 3:
            if depth == 0:
                print_utils.logwarn(f"Invalid depth at pixel ({x}, {y}), skipping this keypoint.")
                continue
        depth_point = rs.rs2_deproject_pixel_to_point(color_intrinsics, [x, y], depth)
        keypoints_vector_in_m.append([depth_point[0], -depth_point[1], depth_point[2]])

    return np.array(keypoints_vector_in_m).flatten()


def pixelToMetric(pixels):
    if isinstance(pixels, tuple):
        pixels = [pixels]
    coords = []
    for (u, v) in pixels:
        x = u * PIXEL_TO_M_X
        y = v * PIXEL_TO_M_Y
        coords.append((x, y))
    return coords


def transformRobotToImage(coords, tf_matrix, dim=2):
    ret = []

    for x, y, z in coords:
        point = np.array([x, y, z, 1])
        new_point = np.linalg.inv(tf_matrix) @ point
        x_new, y_new, z_new, _ = new_point
        ret.append((x_new, y_new, z_new))

    return np.array(ret)


def transformImageToRobot(pixel_coords, tf_matrix, dim=2):
    ret = []
    for x, y, z in pixel_coords:
        point = np.array([x, y, z, 1])
        new_point = tf_matrix @ point
        x_new, y_new, z_new, _ = new_point
        ret.append((x_new, y_new, z_new))
    return np.array(ret)


def transformRobot(pose, tf_matrix, dim=2):
    x, y, z, tx, ty, tz = pose
    point = np.array([x, y, z, 1])
    new_point = tf_matrix @ point
    x_new, y_new, z_new, _ = new_point
    return np.array([x_new, y_new, z_new, tx, ty, tz])


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

    right_eff = None
    model_name = args.controller_control_law

    try:
        dlo = dloDetector("C:\\Users\\User\\pythonProject\\DLO_control\\ws_dlo\\src\\dlo_manipulation_pkg\\scripts\\utils\\config_tf.json")
        right_eff = URrobot(robot_ip_right)
        controller = Controller()
        run_start_time = None

        keypoints_matrix = []

        if not right_eff.is_online:
            print_utils.logerr('Either Robot is not online!')
            return

        right_eff.pointToPointMove(POSE_START2, VEL, ACC)

        print_utils.loginfo("Starting realsense camera...")
        pipeline.start(config)
        align = rs.align(rs.stream.color)
        print_utils.loginfo("camera started!")
        pipeline_started = True

        time_start = time.time()
        send_control_starttime = CONTROL_INTERVAL

        while True:
            print_utils.loginfo(f"Program Interval: {int((time.time() - time_start) * 1000)}ms")
            time_start = time.time()

            frame = pipeline.wait_for_frames()
            aligned_frames = align.process(frame)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

            if not aligned_depth_frame or not color_frame:
                print_utils.logwarn("No frame received from realsense")
                continue

            img = np.asanyarray(color_frame.get_data())
            img_raw = img.copy()

            if frames_to_skip:
                frames_to_skip -= 1
                cv2.putText(img, f"Loading... please wait.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.imshow(img_name, img)
                cv2.waitKey(1)
                continue

            labels = np.array(target_idx)[~np.isin(target_idx, [0, 9])]
            img_with_fp, keypoints_vector, keypoints_matrix, missing_idx = dlo.detectFeatures(
                img, keypoints_matrix, dim=2, labels=labels, nolabel=False
            )

            keypoints_vector_in_m = transformImageToRobot(
                keypoints_vector, TRANSFORM_MATRIX["tf_img_rb1"], dim=dim
            )

            print_utils.loginfo(f"\nkeypoints in m: {keypoints_vector_in_m}")

            img_fp_and_region = drawRegionOnImage(
                img_with_fp,
                transformRobotToImage(
                    desired_pose[case_id],
                    TRANSFORM_MATRIX["tf_img_rb1"],
                    dim=dim
                ),
                region=None
            )

            img_fp_and_region = dlo.showDetectionRegion(img_fp_and_region)
            cv2.putText(img_fp_and_region, f"case {case_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.imshow(img_name, img_fp_and_region)

            if video_writer is not None:
                video_writer.write(img_fp_and_region)

            if first_instance:
                print_utils.loginfo("First instance, initializing camera matrix...")
                prev_keypoints_vector = keypoints_vector_in_m
                first_instance = False

                if not np.any(keypoints_vector):
                    print_utils.logwarn("No keypoints detected, retrying...")
                    keypoints_matrix = []
                    if cv2.waitKey(1) == ord('q'):
                        print_utils.loginfo("Exiting program...")
                        break
                    first_instance = True
                continue

            state = getState(
                dlo_length,
                None, right_eff,
                keypoints_vector_in_m, prev_keypoints_vector,
                desired_pose[case_id],
                dim=dim
            )

            prev_keypoints_vector = keypoints_vector_in_m
            state[I.left_end_avel_idx + I.right_end_avel_idx] /= 2 * np.pi

            control_input = controller.generateControlInput(state)
            print(control_input)

            control_input[[3, 4, 5, 9, 10, 11]] *= 2 * np.pi

            print(f"Robot Control Input: {control_input}")

            if run_start_time is not None:
                desired_points_case = desired_pose[case_id].copy()

                current_points = getCurrentFeaturePoints(
                    right_eff=right_eff,
                    keypoints_vector_in_m=keypoints_vector_in_m,
                    desired_pose_case=desired_points_case,
                    dim=dim
                )

                t_case = time.time() - run_start_time

                log_feature_metrics(
                    robot=right_eff,
                    t=t_case,
                    model_name=model_name,
                    case_id=case_id,
                    desired_points=desired_points_case,
                    current_points=current_points,
                    control_input=control_input,
                    missing_idx=missing_idx
                )

            print(
                "'ENTER' to start move\n"
                "'SPACE' to pause \n"
                "'x' to move one step\n"
                "'q' to quit, \n"
                "'r' to refresh keypoints, \n"
                "'n' to next iteration\n\n"
            )

            if keep_going:
                key = cv2.waitKey(1)
                if time.time() - jog_starttime > JOG_TIMEOUT:
                    print_utils.loginfo(f"Jog Stopped, timeout: {JOG_TIMEOUT:.3f}")
                    keep_going = False
                    if velocity_mode == 'jog':
                        velocity_mode = None
                        right_eff.velocityJogControlStop()
            else:
                keep_going = False
                key = cv2.waitKey(0)

            if key == ord('\r') and keep_going is False:
                print_utils.logwarn("Robot will commence motion, please ensure the workspace is clear! Press enter again to confirm...")
                while cv2.waitKey(0) != ord('\r'):
                    pass
                print_utils.logwarn("Robot moving, keep clear of the workspace!")
                time.sleep(2)

                velocity_mode = 'jog'
                keep_going = True
                jog_starttime = time.time()
                right_eff.logs = []
                run_start_time = time.time()

                if video_writer is None:
                    video_writer = startVideoWriter(project_dir, env_dim, args.controller_control_law, case_id)

            elif key == ord('x'):
                print_utils.loginfo("Moving one step...")
                keep_going = False
                velocity_mode = 'step'

                if run_start_time is None:
                    right_eff.logs = []
                    run_start_time = time.time()

            elif key == ord(' '):
                print_utils.loginfo("Pausing robot motion...")
                keep_going = False
                if velocity_mode == 'jog':
                    velocity_mode = None
                    right_eff.velocityJogControlStop()

            elif key == ord('q'):
                print_utils.loginfo("Exiting program...")
                if velocity_mode == 'jog':
                    velocity_mode = None
                    right_eff.velocityJogControlStop()

                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    print_utils.loginfo(f"Video saved for case {case_id}")

                controller.reset(state)

                if run_start_time is not None and len(right_eff.logs) > 0:
                    log_dir = os.path.join(project_dir, "results", "real", "logs", model_name, env_dim)
                    os.makedirs(log_dir, exist_ok=True)
                    right_eff.save_logs(os.path.join(log_dir, f"case_{case_id}_partial.csv"))
                    right_eff.logs = []

                break

            elif key == ord('r'):
                print_utils.loginfo("Refreshing keypoints...")
                keep_going = False
                keypoints_matrix = []
                first_instance = True
                if velocity_mode == 'jog':
                    velocity_mode = None
                    right_eff.velocityJogControlStop()

            elif key == ord('n'):
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    print_utils.loginfo(f"Video saved for case {case_id}")

                if run_start_time is not None and len(right_eff.logs) > 0:
                    log_dir = os.path.join(project_dir, "results", "real", "logs", model_name, env_dim)
                    os.makedirs(log_dir, exist_ok=True)
                    right_eff.save_logs(os.path.join(log_dir, f"case_{case_id}.csv"))
                    right_eff.logs = []

                run_start_time = None

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
                    right_eff.velocityJogControlStop()

                right_eff.pointToPointMove(POSE_START2, VEL, ACC)

            if velocity_mode == 'step':
                right_eff.velocityControl(control_input[6:], ACC, DURATION)
                velocity_mode = None

            elif velocity_mode == 'jog':
                while (time.time() - send_control_starttime < CONTROL_INTERVAL):
                    continue
                print_utils.loginfo(f"Control Interval: {int((time.time() - send_control_starttime) * 1000)}ms")
                right_eff.velocityJogControl(control_input[6:], ACC, timeout=2)
                send_control_starttime = time.time()

    except Exception as e:
        tb = sys.exc_info()[2]
        print_utils.logerr(f"LINE {tb.tb_lineno}: {e}")

    finally:
        if pipeline_started:
            pipeline.stop()

        if video_writer is not None:
            video_writer.release()

        if right_eff is not None and len(right_eff.logs) > 0:
            log_dir = os.path.join(project_dir, "results", "real", "logs", model_name, env_dim)
            os.makedirs(log_dir, exist_ok=True)
            right_eff.save_logs(os.path.join(log_dir, f"case_{case_id}_autosave.csv"))
            right_eff.logs = []

        if right_eff is not None:
            right_eff.terminate()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    print_utils.loginfo('Start control script')
    main()
    print_utils.loginfo('End of script')
