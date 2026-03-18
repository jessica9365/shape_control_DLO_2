#!/usr/bin/env python

import os
import time
import json
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
from matplotlib import pyplot as plt
import sys
import rospy
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
import gymnasium as gym
from gym_unity.envs import UnityToGymWrapper
from utils.state_index import I
from utils.utils_missing import create_miss_rectangle, draw_image_w_miss, interpolate_missing_points
from utils.data_utils import to_pixels, concat_np_pos
import cv2
import pandas as pd

control_method = rospy.get_param("controller/control_law")
from controller_ours_v1 import Controller
import torch


def compute_feature_metrics(desired_points, current_points):
    err = desired_points - current_points
    err_norms = np.linalg.norm(err, axis=1)

    out = {
        "rmse_all": float(np.sqrt(np.mean(err ** 2))),
        "mean_err": float(np.mean(err_norms)),
        "max_err": float(np.max(err_norms)),
        "endpoint_mean_err": float(np.mean(err_norms[[0, 9]])),
        "inner_mean_err": float(np.mean(err_norms[1:9])),
        "err": err,
        "err_norms": err_norms,
    }
    return out


def append_log(logs, t, model_name, case_id, desired_points, current_points,
               control_input, missing_idx, prefix="gt"):
    metrics = compute_feature_metrics(desired_points, current_points)
    err = metrics["err"]
    err_norms = metrics["err_norms"]

    row = {
        "t": float(t),
        "model": str(model_name),
        "case": int(case_id),
        "missing_count": int(len(missing_idx) if missing_idx is not None else 0),
        f"{prefix}_rmse_all": float(metrics["rmse_all"]),
        f"{prefix}_mean_err": float(metrics["mean_err"]),
        f"{prefix}_max_err": float(metrics["max_err"]),
        f"{prefix}_endpoint_mean_err": float(metrics["endpoint_mean_err"]),
        f"{prefix}_inner_mean_err": float(metrics["inner_mean_err"]),
    }

    for i in range(10):
        row[f"des_{i}_x"] = float(desired_points[i, 0])
        row[f"des_{i}_y"] = float(desired_points[i, 1])
        row[f"des_{i}_z"] = float(desired_points[i, 2])

        row[f"{prefix}_cur_{i}_x"] = float(current_points[i, 0])
        row[f"{prefix}_cur_{i}_y"] = float(current_points[i, 1])
        row[f"{prefix}_cur_{i}_z"] = float(current_points[i, 2])

        row[f"{prefix}_err_{i}_x"] = float(err[i, 0])
        row[f"{prefix}_err_{i}_y"] = float(err[i, 1])
        row[f"{prefix}_err_{i}_z"] = float(err[i, 2])
        row[f"{prefix}_err_{i}_norm"] = float(err_norms[i])

    for j in range(len(control_input)):
        row[f"u_{j}"] = float(control_input[j])

    logs.append(row)


def merge_log_rows(gt_row, obs_row):
    merged = gt_row.copy()
    for k, v in obs_row.items():
        if k not in merged:
            merged[k] = v
    return merged


def save_logs_csv(logs, save_path):
    if len(logs) == 0:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.DataFrame(logs).to_csv(save_path, index=False)
    print(f"Saved log: {save_path}")


class Environment(object):
    def __init__(self):
        self.project_dir = rospy.get_param("project_dir")
        self.env_dim = rospy.get_param("env/dimension")
        self.num_fps = rospy.get_param("DLO/num_FPs")
        self.model_name = control_method

        engine_config_channel = EngineConfigurationChannel()
        env_params_channel = EnvironmentParametersChannel()

        env_file = self.project_dir + "env_dlo/env_" + self.env_dim
        unity_env = UnityEnvironment(
            file_name=env_file,
            seed=1,
            side_channels=[engine_config_channel, env_params_channel]
        )
        engine_config_channel.set_configuration_parameters(
            width=1280,
            height=720,
            time_scale=4.0
        )

        self.env = UnityToGymWrapper(unity_env)
        self.controller = Controller()
        self.control_input = np.zeros((12, ))

        self.logs = []
        self.rollout_start_time = None
        self.rollout_idx = 0

    def get_log_dir(self):
        return os.path.join(
            self.project_dir,
            "results",
            "sim",
            "logs",
            self.model_name,
            self.env_dim
        )

    def start_new_rollout_log(self):
        self.logs = []
        self.rollout_start_time = time.time()

    def save_current_rollout_log(self, suffix=""):
        if len(self.logs) == 0:
            return
        fname = f"case_{self.rollout_idx}{suffix}.csv"
        save_path = os.path.join(self.get_log_dir(), fname)
        save_logs_csv(self.logs, save_path)

    def mainLoop(self):
        for k in range(10):
            state, reward, done, _ = self.env.step(self.control_input)
            state[I.left_end_avel_idx + I.right_end_avel_idx] /= 2 * np.pi

        last_pred_fp_pos = state[I.state_input_idx][3:27].reshape(8, 3).copy()
        observed_state = state.copy()
        missing_idx = []

        self.start_new_rollout_log()

        while not rospy.is_shutdown():
            target_pos = state[I.desired_pos_idx].reshape(10, 3).copy()

            self.control_input = self.controller.generateControlInput(state, observed_state).copy()
            # self.control_input = self.controller.generateControlInput(observed_state).copy()
            self.control_input[[3, 4, 5, 9, 10, 11]] *= 2 * np.pi

            state, reward, done, _ = self.env.step(self.control_input)
            state[I.left_end_avel_idx + I.right_end_avel_idx] /= 2 * np.pi

            state_input = state[I.state_input_idx]
            fp_pos = state_input[3:27].reshape(8, 3)
            all_fp_pos = state_input[0:30].reshape(10, 3)
            observed_fp_pos = fp_pos.copy()
            observed_state = state.copy()
            bbox_coords = []

            # missing_idx, bbox_coords = create_miss_rectangle(target_pos[:, :2], fp_pos[:, :2], width=0.07, height=0.07)
            # missing_idx = [0, 2, 4, 6]
            print("missing_idx", missing_idx)

            all_fp_pos_interp = interpolate_missing_points(
                all_fp_pos,
                [x + 1 for x in missing_idx],
                inplace=False
            )

            for idx in missing_idx:
                observed_fp_pos[idx, :] = all_fp_pos_interp[idx + 1, :]

            delta_t = np.array(0.1)
            observed_state[4:28] = observed_fp_pos.reshape(-1)
            observed_state[48:72] = ((observed_fp_pos - last_pred_fp_pos) / delta_t).reshape(-1)

            last_pred_fp_pos = observed_fp_pos.copy()

            # -------- logging --------
            if self.rollout_start_time is not None:
                t_case = time.time() - self.rollout_start_time

                gt_points = all_fp_pos.copy()

                obs_points = all_fp_pos.copy()
                obs_points[1:9] = observed_fp_pos

                gt_tmp = []
                obs_tmp = []

                append_log(
                    gt_tmp, t_case, self.model_name, self.rollout_idx,
                    target_pos, gt_points, self.control_input, missing_idx, prefix="gt"
                )
                append_log(
                    obs_tmp, t_case, self.model_name, self.rollout_idx,
                    target_pos, obs_points, self.control_input, missing_idx, prefix="obs"
                )

                merged_row = merge_log_rows(gt_tmp[0], obs_tmp[0])
                self.logs.append(merged_row)

            # -------- visualize --------
            end_pos_data = state_input[30:].reshape(2, 7)[:, [0, 1, 5]]
            x_true, y_true = to_pixels(
                torch.tensor(concat_np_pos(fp_pos[:, :2], end_pos_data[:, :2])),
                1920, 1080, x_range=(-1, 1), y_range=(-1, 1)
            )
            x_target, y_target = to_pixels(
                torch.tensor(target_pos)[1:-1, :2],
                1920, 1080, x_range=(-1, 1), y_range=(-1, 1)
            )
            x_miss, y_miss = to_pixels(
                torch.tensor(concat_np_pos(observed_fp_pos[:, :2], end_pos_data[:, :2])),
                1920, 1080, x_range=(-1, 1), y_range=(-1, 1)
            )

            image = draw_image_w_miss(
                x_true, y_true, x_miss, y_miss,
                x_target, y_target, missing_idx, bbox_coords, num_fp=8
            )
            text = f"Rollout {self.rollout_idx}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, text, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("Rollout", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.save_current_rollout_log("_partial")
                break
            elif key == ord('n'):
                done = True

            if done:
                self.save_current_rollout_log()

                self.controller.reset(state)
                state = self.env.reset()

                last_pred_fp_pos = state[I.state_input_idx][3:27].reshape(8, 3).copy()
                observed_state = state.copy()

                self.rollout_idx += 1
                self.start_new_rollout_log()


if __name__ == '__main__':
    try:
        rospy.init_node("sim_env_node")
        env = Environment()
        env.mainLoop()

    except rospy.ROSInterruptException:
        print("program interrupted before completion.")
