#!/usr/bin/env python

# main node of the simulation
# bridge between the Unity Simulator and the controller scripts, based on 'mlagents' and 'gym'

import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
from matplotlib import pyplot as plt
import time
import sys
import rospy
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
import gymnasium as gym
# import gym
from gym_unity.envs import UnityToGymWrapper
from utils.state_index import I
#from geometry_msgs.msg import Vector3
from utils.utils_missing import create_miss_rectangle, draw_image_w_miss, interpolate_missing_points
from utils.data_utils import to_pixels, concat_np_pos
import cv2

control_method = rospy.get_param("controller/control_law")
# if control_method == 'ours':
#     # from controller_ours import Controller
#     # from controller_ours_offline import Controller
#     # from controller_ours_NN_graph import Controller
#     # from controller_ours_offline_GNN import Controller
from controller_ours_v2_v3 import Controller
import torch

class Environment(object):
    def __init__(self):
        self.project_dir = rospy.get_param("project_dir")
        env_dim = rospy.get_param("env/dimension")
        self.num_fps = rospy.get_param("DLO/num_FPs")

        engine_config_channel = EngineConfigurationChannel()
        env_params_channel = EnvironmentParametersChannel()

        # use the built Unity environment
        env_file = self.project_dir + "env_dlo/env_" + env_dim
        unity_env = UnityEnvironment(file_name=env_file, seed=1, side_channels=[engine_config_channel, env_params_channel])
        engine_config_channel.set_configuration_parameters(width=1280, height=720, time_scale=4.0)  # speed x2

        self.env = UnityToGymWrapper(unity_env)
        self.controller = Controller()
        self.control_input = np.zeros((12, ))

        self.fp_visible_mask = np.ones((self.num_fps,), dtype=np.float32)




    # -------------------------------------------------------------------
    def mainLoop(self):
        # the first second in unity is not stable, so we do nothing in the first second
        for k in range(10):
            state, reward, done, _ = self.env.step(self.control_input)
            state[I.left_end_avel_idx + I.right_end_avel_idx] /= 2*np.pi  # change the unit of the input angular velocity from rad/s  to 2pi*rad/s

        last_pred_fp_pos = state[I.state_input_idx][3:27].reshape(8, 3).copy()
        observed_state = state.copy()
        missing_idx=[]
        fp_visible_mask = np.ones((self.num_fps,), dtype=np.float32)


        rollout_idx = 0
        while not rospy.is_shutdown():
            target_pos = state[I.desired_pos_idx].reshape(10, 3)[:, :2].copy()
            #----------------------------------- original scripts ---------------------------------#

            self.control_input = self.controller.generateControlInput(
                state,
                observed_state=observed_state,
                fp_visible_mask=self.fp_visible_mask
            ).copy()
            self.control_input[[3, 4, 5, 9, 10, 11]] *= 2*np.pi  # change the unit of the output angular velocity from 2pi*rad/s  torad/s

            state, reward, done, _ = self.env.step(self.control_input)
            state[I.left_end_avel_idx + I.right_end_avel_idx] /= 2*np.pi # change the unit of the input angular velocity from rad/s  to 2pi*rad/s
            #----------------------------------- - ---------------------------------#

            #---------------------------------- creat missing idx ---------------------------------#
            state_input = state[I.state_input_idx]
            fp_pos = state_input[3:27].reshape(8, 3)
            all_fp_pos = state_input[0:30].reshape(10, 3)
            observed_fp_pos = fp_pos.copy()
            observed_state = state.copy()
            bbox_coords = []
            # missing_idx, bbox_coords = create_miss_rectangle(target_pos, fp_pos, width=0.07, height=0.07) #occlusion rectangle and coordinates
            missing_idx = [0,2,4,6]
            # missing_idx = [3,4]
            # missing_idx = [1,3,5]
            # missing_idx = [2,3,4,5]
            # missing_idx = [1,2,3,5,6]
            print("missing_idx", missing_idx)

            # --- build visibility mask: 1 visible, 0 occluded ---
            fp_visible_mask = np.ones((self.num_fps,), dtype=np.float32)
            for idx in missing_idx:
                fp_visible_mask[idx + 1] = 0.0   # +1 if your 8 tracked FPs are indices 1..8 of the 10 FPs
            self.fp_visible_mask = fp_visible_mask


            # ------------------------ replace missing with last known values ---------------#
            # observed_fp_pos[missing_idx,:] = last_pred_fp_pos[missing_idx,:]

            #------------------------- linear interpolation ---------------------------------#
            all_fp_pos = interpolate_missing_points(all_fp_pos,
                                           [x+1 for x in missing_idx],
                                           inplace = False)

            for idx in missing_idx:
                observed_fp_pos[idx, :] = all_fp_pos[idx+1, :]


            delta_t = np.array(0.1)
            # --- reshape back and insert corrected slice ---
            # first index is length, then are state_input
            # [3:27] +1 --> [4:28]
            observed_state[4:28] = observed_fp_pos.reshape(-1)
            observed_state[48:72] = ((observed_fp_pos- last_pred_fp_pos) / delta_t).reshape(-1)

            # --- update last known values ---
            last_pred_fp_pos = observed_fp_pos.copy()

            # #--------------------------------------- Visualize ----------------------- #
            # #ground truth pos
            end_pos_data = state_input[30:].reshape(2, 7)[:, [0, 1, 5]]
            x_true, y_true = to_pixels(torch.tensor(concat_np_pos(fp_pos[:,:2], end_pos_data[:,:2])[1:-1, :2]), 1920,1080, x_range=(-1, 1), y_range=(-1, 1))
            x_true, y_true = to_pixels(torch.tensor(concat_np_pos(fp_pos[:,:2], end_pos_data[:,:2])), 1920,1080, x_range=(-1, 1), y_range=(-1, 1))
            x_target, y_target = to_pixels(torch.tensor(target_pos)[1:-1,:2], 1920,1080,
                                       x_range=(-1, 1), y_range=(-1, 1))

            x_miss, y_miss = to_pixels(torch.tensor(concat_np_pos(observed_fp_pos[:,:2], end_pos_data[:,:2])), 1920,1080,
                                       x_range=(-1, 1), y_range=(-1, 1))

            image = draw_image_w_miss (x_true, y_true, x_miss, y_miss, x_target, y_target, missing_idx, bbox_coords, num_fp=8)
            text = f"Rollout {rollout_idx}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, text, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("Rollout", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                done = True

            if done: # Time up (30s), the env and the controller are reset. Next case with different desired shapes.
                self.controller.reset(state)
                state = self.env.reset()

                last_pred_fp_pos = state[I.state_input_idx][3:27].reshape(8, 3).copy()
                observed_state = state.copy()
                obs_idx = [0, 1, 2, 3, 4, 5, 6, 7]

                rollout_idx += 1


# --------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        rospy.init_node("sim_env_node")
        env = Environment()
        env.mainLoop()

    except rospy.ROSInterruptException:
        print("program interrupted before completion.")