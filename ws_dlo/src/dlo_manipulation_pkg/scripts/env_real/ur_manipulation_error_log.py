#!/usr/bin/env python3
from ur_controller import URrobot
from controller_ours_rbf import Controller  # change the controller script
# from controller_ours_GNN_NN import Controller
# from controller_ours_GNN_NN_region import Controller
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
import pandas as pd


# ─────────────────────────────────────────────
# RealSense pipeline
# ─────────────────────────────────────────────
pipeline = rs.pipeline()
rs_config = rs.config()
rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)


# ─────────────────────────────────────────────
# Robot IPs
# ─────────────────────────────────────────────
VMWARE_URSIM_IP1  = '172.16.45.129'
VMWARE_URSIM_IP2  = '172.16.45.128'
NTU_UR5E_MAIN_IP  = '10.149.230.1'
NTU_UR5E_LOAN_IP  = '10.149.230.2'

robot_ip_right = NTU_UR5E_MAIN_IP


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
dim           = 2
PIXEL_TO_M_X  = 0.0016
PIXEL_TO_M_Y  = 0.0016
ACC           = 0.1
VEL           = 0.03
DURATION      = 0.1 * 5
JOG_TIMEOUT   = 30
CONTROL_INTERVAL = 0.1
IMG_NAME      = "realsense_camera"

TRANSFORM_MATRIX = {
    "tf_img_rb1": args_ur.tf_img_rb1_2D_onearm,
    "tf_rb2_rb1": args_ur.tf_rb2_rb1_2D_onearm,
}
POSE_START2  = args_ur.pose_start2_2D_onearm
dlo_length   = args_ur.dlo_length

project_dir  = args.project_dir
target_idx   = args.controller_object_fps_idx
env_dim      = args.env_dimension

desired_pos_file = (
    project_dir
    + "/env_dlo/env_" + env_dim + "_Data/StreamingAssets/desired_shape/"
    + env_dim + "/" + "realsim_desired_positions_onearm.npy"
)
desired_pose  = np.load(desired_pos_file).reshape(-1, 10, 3)
nr_of_cases   = desired_pose.shape[0]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_log_dir(model_name):
    return os.path.join(project_dir, "results", "real", "logs", model_name, env_dim)


def transformRobot(pose, tf_matrix, dim=2):
    x, y, z, tx, ty, tz = pose
    new_point = tf_matrix @ np.array([x, y, z, 1])
    x_new, y_new, z_new, _ = new_point
    return np.array([x_new, y_new, z_new, tx, ty, tz])


def transformRobotToImage(coords, tf_matrix, dim=2):
    ret = []
    for x, y, z in coords:
        new_point = np.linalg.inv(tf_matrix) @ np.array([x, y, z, 1])
        ret.append(tuple(new_point[:3]))
    return np.array(ret)


def transformImageToRobot(pixel_coords, tf_matrix, dim=2):
    ret = []
    for x, y, z in pixel_coords:
        new_point = tf_matrix @ np.array([x, y, z, 1])
        ret.append(tuple(new_point[:3]))
    return np.array(ret)


def getState(dlo_length, left_eff, right_eff, keypoints_vector,
             prev_keypoints_vector, desired_pose_case, dim=2):
    """
    117-dim state vector.
    0        : DLO length
    1..30    : 10 feature point positions (10×3)
    31..44   : end-effector poses (2×7 → positions + quats)
    45..74   : feature point velocities (10×3)
    75..86   : end-effector velocities (2×6)
    87..116  : desired feature positions (10×3)
    """
    state = np.zeros(117)
    keypoints_vel = keypoints_vector - prev_keypoints_vector

    left_eff_pose  = desired_pose_case.flatten()[0:3].copy()
    right_eff_pose = transformRobot(
        right_eff.receive.getActualTCPPose(),
        TRANSFORM_MATRIX["tf_rb2_rb1"], dim=dim
    )

    if dim == 2:
        left_eff_pose[2]  = 0
        right_eff_pose[2] = 0

    left_eff_quat  = [0, 0, 0, 0]
    left_eff_vel   = [0, 0, 0, 0, 0, 0]
    right_eff_quat = R.from_euler('xyz', right_eff_pose[3:]).as_quat()
    right_eff_vel  = np.array(right_eff.receive.getActualTCPSpeed())

    state[0]      = dlo_length
    state[1:4]    = left_eff_pose[:3]
    state[4:28]   = keypoints_vector.flatten()
    state[28:31]  = right_eff_pose[:3]
    state[31:34]  = left_eff_pose[:3]
    state[34:38]  = left_eff_quat
    state[38:41]  = right_eff_pose[:3]
    state[41:45]  = right_eff_quat
    state[45:48]  = left_eff_vel[:3]
    state[48:72]  = keypoints_vel.flatten()
    state[72:75]  = right_eff_vel[:3]
    state[75:81]  = left_eff_vel
    state[81:87]  = right_eff_vel
    state[87:117] = desired_pose_case.flatten()

    return state.astype(np.float32)


def getCurrentFeaturePoints(right_eff, keypoints_vector_in_m, desired_pose_case, dim=2):
    """
    Returns (10, 3) array:
      - point 0   : left endpoint  (approximated from desired pose, one-arm setup)
      - points 1–8: 8 detected internal keypoints
      - point 9   : right endpoint (from robot TCP)
    """
    current_points = np.zeros((10, 3), dtype=np.float32)

    left_eff_pose  = desired_pose_case[0].copy()
    right_eff_pose = transformRobot(
        right_eff.receive.getActualTCPPose(),
        TRANSFORM_MATRIX["tf_rb2_rb1"], dim=dim
    )

    if dim == 2:
        left_eff_pose[2]  = 0
        right_eff_pose[2] = 0

    kp = keypoints_vector_in_m.reshape(-1, 3)
    if kp.shape[0] != 8:
        raise ValueError(f"Expected 8 internal keypoints, got {kp.shape[0]}")

    current_points[0]   = left_eff_pose[:3]
    current_points[1:9] = kp
    current_points[9]   = right_eff_pose[:3]

    return current_points


def build_visibility_mask(missing_idx):
    """
    missing_idx refers to the 8 internal keypoints (0-indexed, range 0–7).
    Converts to a 10-point boolean mask:
      - Index 0  (left endpoint)  : always visible (from desired pose)
      - Indices 1–8 (internal kps): depends on missing_idx
      - Index 9  (right endpoint) : always visible (from TCP)

    FIX vs old code: old code built mask directly from missing_idx as if
    it mapped to all 10 points. This is now correctly shifted by +1.
    """
    mask = np.ones(10, dtype=bool)
    for idx in (missing_idx or []):
        if 0 <= idx < 8:
            mask[idx + 1] = False   # shift: internal pt 0 → global pt 1
    return mask


def log_feature_metrics(robot, t, model_name, case_id, desired_points,
                        current_points, control_input, missing_idx,
                        prefix="trajectory"):
    """
    Logs one timestep of feature-point error.

    prefix="trajectory" : regular control step (some points may be occluded)
    prefix="final_gt"   : final clean frame after occlusion removed (all visible)

    FIX vs old code:
      - Restores full per-point logging (des_i_*, cur_i_*, err_i_*)
      - Restores control input logging (u_j)
      - Restores TCP pose logging
      - Adds prefix, n_visible, rmse_visible, mean_visible
      - Adds is_visible_i columns (one per point)
      - Correctly builds 10-pt visibility mask from 8-pt missing_idx
    """
    visible_mask = build_visibility_mask(missing_idx)
    visible_idx  = np.where(visible_mask)[0]
    n_visible    = int(visible_mask.sum())

    err       = desired_points - current_points
    err_norms = np.linalg.norm(err, axis=1)

    # Visible-only metrics (meaningful during occluded trajectory)
    if prefix == "trajectory" and n_visible > 0:
        rmse_visible = float(np.sqrt(np.mean(err[visible_idx] ** 2)))
        mean_visible = float(np.mean(err_norms[visible_idx]))
    else:
        rmse_visible = np.nan
        mean_visible = np.nan

    # All-points metrics (valid for final_gt; available but use with caution
    # during occluded trajectory since some cur_i values are interpolated)
    rmse_all          = float(np.sqrt(np.mean(err ** 2)))
    mean_err          = float(np.mean(err_norms))
    max_err           = float(np.max(err_norms))
    endpoint_mean_err = float(np.mean(err_norms[[0, 9]]))
    inner_mean_err    = float(np.mean(err_norms[1:9]))

    tcp_pose = transformRobot(
        robot.receive.getActualTCPPose(),
        TRANSFORM_MATRIX["tf_rb2_rb1"], dim=dim
    )

    row = {
        "t":                  float(t),
        "model":              str(model_name),
        "case":               int(case_id),
        "prefix":             prefix,
        "missing_count":      int(len(missing_idx) if missing_idx else 0),
        "n_visible":          n_visible,
        # Trajectory-safe metrics (visible points only)
        "rmse_visible":       rmse_visible,
        "mean_visible":       mean_visible,
        # All-points metrics (ground truth quality for final_gt rows)
        "rmse_all":           rmse_all,
        "mean_err":           mean_err,
        "max_err":            max_err,
        "endpoint_mean_err":  endpoint_mean_err,
        "inner_mean_err":     inner_mean_err,
        # TCP pose
        "tcp_x":              float(tcp_pose[0]),
        "tcp_y":              float(tcp_pose[1]),
        "tcp_z":              float(tcp_pose[2]),
    }

    # Per-point fields (used for per-point error charts + occlusion masking)
    for i in range(10):
        row[f"is_visible_{i}"] = int(visible_mask[i])   # 1 = visible, 0 = occluded
        row[f"des_{i}_x"]      = float(desired_points[i, 0])
        row[f"des_{i}_y"]      = float(desired_points[i, 1])
        row[f"des_{i}_z"]      = float(desired_points[i, 2])
        row[f"cur_{i}_x"]      = float(current_points[i, 0])
        row[f"cur_{i}_y"]      = float(current_points[i, 1])
        row[f"cur_{i}_z"]      = float(current_points[i, 2])
        row[f"err_{i}_x"]      = float(err[i, 0])
        row[f"err_{i}_y"]      = float(err[i, 1])
        row[f"err_{i}_z"]      = float(err[i, 2])
        row[f"err_{i}_norm"]   = float(err_norms[i])

    # Control inputs
    for j in range(len(control_input)):
        row[f"u_{j}"] = float(control_input[j])

    robot.logs.append(row)


def capture_frame(pipeline, align):
    """
    Grabs one fresh aligned RealSense frame.
    Returns (img_raw, aligned_depth_frame, color_intrinsics) or None on failure.

    FIX: Used in final_gt branch so the clean frame is genuinely new,
    not reusing stale keypoints from the last occluded control step.
    """
    for _ in range(10):         # up to 10 attempts
        frame = pipeline.wait_for_frames(timeout_ms=3000)
        aligned = align.process(frame)
        depth   = aligned.get_depth_frame()
        color   = aligned.get_color_frame()
        if depth and color:
            img = np.asanyarray(color.get_data())
            intrinsics = color.profile.as_video_stream_profile().intrinsics
            return img, depth, intrinsics
    return None, None, None


def drawRegionOnImage(img, desired_pose_px, region=None):
    if region is None:
        region = [0.0] * len(desired_pose_px)
    for idx, ((x, y, z), r) in enumerate(zip(desired_pose_px, region)):
        if idx in target_idx:
            thickness = 1
            r /= np.mean([PIXEL_TO_M_X, PIXEL_TO_M_Y])
            if r == 0:
                thickness = -1
                r = 3
            cv2.circle(img, (int(x), int(y)), int(r), (0, 255, 0), thickness)
    return img


def rsGetMetric(keypoints_vector, aligned_depth_frame, color_intrinsics, dim=2):
    out = []
    for (x, y, z) in keypoints_vector:
        depth = aligned_depth_frame.get_distance(int(x), int(y))
        if dim == 3 and depth == 0:
            print_utils.logwarn(f"Invalid depth at ({x},{y}), skipping.")
            continue
        pt = rs.rs2_deproject_pixel_to_point(color_intrinsics, [x, y], depth)
        out.append([pt[0], -pt[1], pt[2]])
    return np.array(out).flatten()


def startVideoWriter(case_id, model_name):
    result_dir = os.path.join(project_dir, "results", "real", "control", model_name, env_dim)
    os.makedirs(result_dir, exist_ok=True)
    path   = os.path.join(result_dir, f"video_case_{case_id}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, 10.0, (640, 480))
    print_utils.loginfo(f"Recording: {path}")
    return writer


def reset_for_next_case(right_eff, controller, state, velocity_mode):
    """
    Centralises all the reset actions needed between cases.

    FIX: Old code had partial/inconsistent resets scattered across key handlers.
    Now all state is fully reset in one place.
    """
    if velocity_mode == 'jog':
        right_eff.velocityJogControlStop()
    controller.reset(state)
    right_eff.pointToPointMove(POSE_START2, VEL, ACC)
    return None, False, False, True, [], None   # velocity_mode, keep_going, jog vars, first_instance, keypoints_matrix, run_start_time


def save_trajectory(right_eff, model_name, case_id):
    """Saves only trajectory-prefix rows to case_<id>_trajectory.csv."""
    rows = [r for r in right_eff.logs if r.get("prefix") == "trajectory"]
    if not rows:
        print_utils.logwarn(f"No trajectory rows to save for case {case_id}.")
        return
    log_dir = get_log_dir(model_name)
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"case_{case_id}_trajectory.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print_utils.loginfo(f"✓ Trajectory saved: {path}")


def save_final_gt(right_eff, model_name, case_id):
    """Saves only final_gt-prefix rows to case_<id>_final_gt.csv."""
    rows = [r for r in right_eff.logs if r.get("prefix") == "final_gt"]
    if not rows:
        print_utils.logwarn(f"No final_gt rows to save for case {case_id}.")
        return
    log_dir = get_log_dir(model_name)
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"case_{case_id}_final_gt.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print_utils.loginfo(f"✓ Final GT saved: {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    pipeline_started     = False
    first_instance       = True
    frames_to_skip       = 50
    keep_going           = False
    velocity_mode        = None
    jog_starttime        = None
    case_id              = 0
    video_writer         = None
    run_start_time       = None
    right_eff            = None
    model_name           = args.controller_control_law
    send_control_starttime = CONTROL_INTERVAL

    # ── Initialise ────────────────────────────
    try:
        dlo = dloDetector(
            "C:\\Users\\User\\pythonProject\\DLO_control\\"
            "ws_dlo\\src\\dlo_manipulation_pkg\\scripts\\utils\\config_tf.json"
        )
        right_eff  = URrobot(robot_ip_right)
        controller = Controller()

        if not right_eff.is_online:
            print_utils.logerr("Robot is not online!")
            return

        right_eff.pointToPointMove(POSE_START2, VEL, ACC)

        print_utils.loginfo("Starting RealSense camera...")
        pipeline.start(rs_config)
        align = rs.align(rs.stream.color)
        print_utils.loginfo("Camera started!")
        pipeline_started = True

        keypoints_matrix = []
        time_start = time.time()

        # ── Main loop ─────────────────────────
        while True:
            print_utils.loginfo(f"Interval: {int((time.time() - time_start)*1000)}ms")
            time_start = time.time()

            # ── Grab frame ────────────────────
            frame          = pipeline.wait_for_frames()
            aligned_frames = align.process(frame)
            depth_frame    = aligned_frames.get_depth_frame()
            color_frame    = aligned_frames.get_color_frame()
            color_intr     = color_frame.profile.as_video_stream_profile().intrinsics

            if not depth_frame or not color_frame:
                print_utils.logwarn("No frame received from RealSense")
                continue

            img = np.asanyarray(color_frame.get_data())

            # ── Skip first N frames ───────────
            if frames_to_skip:
                frames_to_skip -= 1
                cv2.putText(img, "Loading... please wait.",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.imshow(IMG_NAME, img)
                cv2.waitKey(1)
                continue

            # ── Detect features ───────────────
            labels = np.array(target_idx)[~np.isin(target_idx, [0, 9])]
            img_with_fp, keypoints_vector, keypoints_matrix, missing_idx = \
                dlo.detectFeatures(img, keypoints_matrix, dim=2, labels=labels, nolabel=False)

            keypoints_vector_in_m = transformImageToRobot(
                keypoints_vector, TRANSFORM_MATRIX["tf_img_rb1"], dim=dim
            )
            print_utils.loginfo(f"Keypoints (m): {keypoints_vector_in_m}")

            # ── Draw desired shape + detection region ──
            img_vis = drawRegionOnImage(
                img_with_fp,
                transformRobotToImage(desired_pose[case_id],
                                      TRANSFORM_MATRIX["tf_img_rb1"], dim=dim),
            )
            img_vis = dlo.showDetectionRegion(img_vis)
            cv2.putText(img_vis, f"Case {case_id} | missing: {len(missing_idx)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.imshow(IMG_NAME, img_vis)

            if video_writer is not None:
                video_writer.write(img_vis)

            # ── First-frame initialisation ────
            if first_instance:
                print_utils.loginfo("Initialising keypoint history...")
                prev_keypoints_vector = keypoints_vector_in_m.copy()
                first_instance = False
                if not np.any(keypoints_vector):
                    print_utils.logwarn("No keypoints, retrying...")
                    keypoints_matrix = []
                    if cv2.waitKey(1) == ord('q'):
                        break
                    first_instance = True
                continue

            # ── Build state ───────────────────
            state = getState(dlo_length, None, right_eff,
                             keypoints_vector_in_m, prev_keypoints_vector,
                             desired_pose[case_id], dim=dim)
            prev_keypoints_vector = keypoints_vector_in_m.copy()
            state[I.left_end_avel_idx + I.right_end_avel_idx] /= 2 * np.pi

            # ── Controller ────────────────────
            control_input = controller.generateControlInput(state, state)
            # control_input = controller.generateControlInput(state)
            control_input[[3, 4, 5, 9, 10, 11]] *= 2 * np.pi
            print_utils.loginfo(f"Control input: {control_input}")

            # ── Log trajectory step ───────────
            if run_start_time is not None:
                desired_pts  = desired_pose[case_id].copy()
                current_pts  = getCurrentFeaturePoints(right_eff,
                                                       keypoints_vector_in_m,
                                                       desired_pts, dim=dim)
                t_case = time.time() - run_start_time

                log_feature_metrics(
                    robot=right_eff, t=t_case,
                    model_name=model_name, case_id=case_id,
                    desired_points=desired_pts, current_points=current_pts,
                    control_input=control_input, missing_idx=missing_idx,
                    prefix="trajectory"
                )

            # ── Key bindings summary ──────────
            print(
                "\n=== CONTROLS ===\n"
                " ENTER : start controller\n"
                " SPACE : pause controller\n"
                " x     : single step\n"
                " n     : END CASE → save trajectory → prompt final GT frame\n"
                " f     : FINAL GT FRAME  (remove occlusion first!)\n"
                " r     : refresh keypoints\n"
                " q     : quit\n"
                "================\n"
            )

            # ── Keyboard input ────────────────
            if keep_going:
                key = cv2.waitKey(1) & 0xFF
                if jog_starttime and time.time() - jog_starttime > JOG_TIMEOUT:
                    print_utils.loginfo(f"Jog timeout ({JOG_TIMEOUT}s)")
                    keep_going    = False
                    velocity_mode = None
                    right_eff.velocityJogControlStop()
            else:
                key = cv2.waitKey(0) & 0xFF

            # ── ENTER: start controller ───────
            if key == ord('\r') and not keep_going:
                print_utils.logwarn("Starting motion — confirm workspace is clear. Press ENTER again.")
                while cv2.waitKey(0) & 0xFF != ord('\r'):
                    pass
                print_utils.logwarn("Robot moving!")
                time.sleep(2)
                velocity_mode  = 'jog'
                keep_going     = True
                jog_starttime  = time.time()
                right_eff.logs = []
                run_start_time = time.time()
                if video_writer is None:
                    video_writer = startVideoWriter(case_id, model_name)

            # ── x: single step ───────────────
            elif key == ord('x'):
                keep_going    = False
                velocity_mode = 'step'
                if run_start_time is None:
                    right_eff.logs = []
                    run_start_time = time.time()

            # ── SPACE: pause ──────────────────
            elif key == ord(' '):
                keep_going = False
                if velocity_mode == 'jog':
                    velocity_mode = None
                    right_eff.velocityJogControlStop()

            # ── r: refresh keypoints ──────────
            elif key == ord('r'):
                keep_going       = False
                keypoints_matrix = []
                first_instance   = True
                if velocity_mode == 'jog':
                    velocity_mode = None
                    right_eff.velocityJogControlStop()

            # ── q: quit ───────────────────────
            elif key == ord('q'):
                if velocity_mode == 'jog':
                    right_eff.velocityJogControlStop()
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                controller.reset(state)
                if run_start_time is not None and right_eff.logs:
                    log_dir = get_log_dir(model_name)
                    os.makedirs(log_dir, exist_ok=True)
                    right_eff.save_logs(
                        os.path.join(log_dir, f"case_{case_id}_partial.csv")
                    )
                break

            # ── n: end case ───────────────────
            elif key == ord('n'):
                # Stop controller
                if velocity_mode == 'jog':
                    right_eff.velocityJogControlStop()
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    print_utils.loginfo(f"Video saved for case {case_id}")

                # Save trajectory
                save_trajectory(right_eff, model_name, case_id)
                right_eff.logs = []     # clear — ready for final_gt row

                # Prompt for final GT frame
                print_utils.loginfo("\n=== FINAL FRAME ===")
                print_utils.loginfo("• Remove the occlusion object from the scene")
                print_utils.loginfo("• Ensure all DLO feature points are visible")
                print_utils.loginfo("• Press 'f' to capture  |  'q' to skip")

                while True:
                    key_f = cv2.waitKey(0) & 0xFF
                    if key_f in (ord('f'), ord('q')):
                        break

                if key_f == ord('f'):
                    # ── Capture a genuinely NEW clean frame ──────────
                    # FIX: old code reused stale keypoints_vector_in_m from
                    # the last occluded control step. Now we grab a fresh
                    # RealSense frame and rerun detection with no occlusion.
                    print_utils.loginfo("Capturing clean frame...")
                    img_final, depth_final, intr_final = capture_frame(pipeline, align)

                    if img_final is not None:
                        _, kp_final, _, missing_final = dlo.detectFeatures(
                            img_final, [], dim=2, labels=labels, nolabel=False
                        )
                        kp_final_m = transformImageToRobot(
                            kp_final, TRANSFORM_MATRIX["tf_img_rb1"], dim=dim
                        )
                        final_pts = getCurrentFeaturePoints(
                            right_eff, kp_final_m,
                            desired_pose[case_id], dim=dim
                        )
                        log_feature_metrics(
                            robot=right_eff, t=-1.0,
                            model_name=model_name, case_id=case_id,
                            desired_points=desired_pose[case_id].copy(),
                            current_points=final_pts,
                            control_input=np.zeros(12),
                            missing_idx=[],             # all points visible
                            prefix="final_gt"
                        )
                        save_final_gt(right_eff, model_name, case_id)
                    else:
                        print_utils.logwarn("Failed to capture clean frame — final GT not saved.")
                else:
                    print_utils.logwarn(f"Final GT skipped for case {case_id}.")

                # ── Full reset for next case ──────────────────────
                # FIX: old code had partial/inconsistent resets scattered
                # in the key handler. Now all state is reset cleanly here.
                right_eff.logs = []
                run_start_time = None
                keep_going     = False
                velocity_mode  = None
                jog_starttime  = None
                first_instance = True
                keypoints_matrix = []
                controller.reset(state)

                case_id += 1
                if case_id >= nr_of_cases:
                    print_utils.loginfo("All cases complete!")
                    break

                print_utils.loginfo(f"\n=== STARTING CASE {case_id} / {nr_of_cases-1} ===")
                right_eff.pointToPointMove(POSE_START2, VEL, ACC)

            # ── Execute motion ────────────────
            if velocity_mode == 'step':
                right_eff.velocityControl(control_input[6:], ACC, DURATION)
                velocity_mode = None

            elif velocity_mode == 'jog':
                while time.time() - send_control_starttime < CONTROL_INTERVAL:
                    pass
                print_utils.loginfo(
                    f"Control interval: {int((time.time()-send_control_starttime)*1000)}ms"
                )
                right_eff.velocityJogControl(control_input[6:], ACC, timeout=2)
                send_control_starttime = time.time()

    except Exception as e:
        tb = sys.exc_info()[2]
        print_utils.logerr(f"Line {tb.tb_lineno}: {e}")

    finally:
        if pipeline_started:
            pipeline.stop()
        if video_writer is not None:
            video_writer.release()
        if right_eff is not None and right_eff.logs:
            log_dir = get_log_dir(model_name)
            os.makedirs(log_dir, exist_ok=True)
            right_eff.save_logs(
                os.path.join(log_dir, f"case_{case_id}_autosave.csv")
            )
        if right_eff is not None:
            right_eff.terminate()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    print_utils.loginfo("Start control script")
    main()
    print_utils.loginfo("End of script")
