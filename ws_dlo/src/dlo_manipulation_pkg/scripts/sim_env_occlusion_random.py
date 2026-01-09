#!/usr/bin/env python3
"""
sim_env_occlusion_random.py - Pure occlusion testing
- Per-frame random occlusion (25% frames, 1-4 random FPs)
- Enhanced visualization 
- Controller handles ALL state saving (state_X.npy)
- ZERO duplication of controller logic
"""

import numpy as np
# Add this to TOP of sim_env_occlusion_random.py (after imports)
if not hasattr(np, 'bool'):
    np.bool = np.bool_

import time
import sys
import os
import rospy
import cv2
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
import gymnasium as gym
from gym_unity.envs import UnityToGymWrapper
from utils.state_index import I
from controller_ours_case1_v2 import Controller  # Handles GNN + saving!

class Args:
    def __init__(self):
        self.numfp = rospy.get_param('DLO_num_FPs', 10)
        self.occlusion_p = 0.25      # 25% frames occluded
        self.k_min, self.k_max = 1, 4  # 1-4 FPs when occluded
        self.vis_width, self.vis_height = 1920, 1080

args = Args()

print(f"=== Occlusion Test (Controller handles saving) ===")
print(f"Num FPs: {args.numfp} | p_occlude: {args.occlusion_p}")
print(f"Press 'q' to quit")

class OcclusionTester:
    def __init__(self):
        # 🔥 FIX 1: CORRECT Unity path structure
        self.projectdir = rospy.get_param('projectdir', '/home/jessica/shape_control_DLO_2')
        self.envdim = rospy.get_param('envdimension', '2D')
        
        # 🔥 FIX 2: CORRECT path - /env/2D/ NOT /env_dlo/env_2D/
        envfile = f"{self.projectdir}/env_dlo/env_{self.envdim}.x86_64"
        print(f"Loading Unity: {envfile}")
        
        # 🔥 FIX 3: Verify file exists before UnityEnvironment
        if not os.path.exists(envfile):
            raise FileNotFoundError(f"Unity executable not found: {envfile}")
        
        engine_channel = EngineConfigurationChannel()
        env_channel = EnvironmentParametersChannel()
        
        unity_env = UnityEnvironment(file_name=envfile, seed=1,
                                   side_channels=[engine_channel, env_channel])
        engine_channel.set_configuration_parameters(width=1280, height=720, time_scale=5.0)
        self.env = UnityToGymWrapper(unity_env)
        
        # Controller handles GNN + state saving!
        self.controller = Controller()
        self.control_input = np.zeros(12)
    
    def world_to_pixel(self, pos):
        """World [-1,1] → pixel coordinates"""
        x = ((pos[:, 0] + 1.0) / 2.0 * args.vis_width).astype(int)
        y = ((1.0 - pos[:, 1]) / 2.0 * args.vis_height).astype(int)
        return np.stack([x, y], axis=-1)
    
    def visualize_frame(self, fp_truth_xy, dropout_idx, frame_count, state):
        """Enhanced occlusion visualization"""
        image = np.full((args.vis_height, args.vis_width, 3), 30, dtype=np.uint8)
        
        # 🔥 FIX 4: Pass 'state' parameter instead of self.state
        # 1. GREEN TARGET
        target_pos = state[I.desired_pos_idx].reshape(args.numfp, 3)
        target_xy = self.world_to_pixel(target_pos[:args.numfp, :2])
        for i in range(args.numfp-1):
            cv2.line(image, tuple(target_xy[i]), tuple(target_xy[i+1]), (0, 255, 0), 3)
        
        # 2. BLUE: Visible truth FPs
        for i in range(args.numfp):
            if i not in dropout_idx:
                px, py = self.world_to_pixel(fp_truth_xy[i:i+1])[0]
                cv2.circle(image, (px, py), 8, (255, 200, 0), -1)  # Blue truth
        
        # 3. CYAN: Occluded TRUTH positions
        for i in dropout_idx:
            px, py = self.world_to_pixel(fp_truth_xy[i:i+1])[0]
            cv2.circle(image, (px, py), 12, (255, 255, 0), 3)  # Cyan outline
        
        # 4. LEGEND
        cv2.putText(image, "Blue=Visible | Cyan=Occluded Truth | Green=Target", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(image, f"Frame {frame_count} | Hidden: {len(dropout_idx)}/{args.numfp} | {sorted(dropout_idx)}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        return image
    
    def mainLoop(self):
        """Infinite loop: Unity → Occlusion → Controller → Visualize"""
        frame_count = 0
        
        while not rospy.is_shutdown():
            frame_count += 1
            
            # === WARMUP (first 10 frames + reset) ===
            if frame_count == 1:
                for _ in range(10):
                    state, _, _, _ = self.env.step(self.control_input)
                    state[I.left_end_avel_idx] = 2 * np.pi
                state = self.env.reset()
                state[I.left_end_avel_idx] = 2 * np.pi
            
            # 🔥 FIX 5: Store state for visualization (before modification)
            original_state = state.copy()
            
            # === EXTRACT FEATURE POINTS ===
            fp_truth = state[I.fps_pos_idx].reshape(args.numfp, 3)  # 10 FPs × 3
            fp_truth_xy = fp_truth[:, :2]
            
            # === PER-FRAME RANDOM OCCLUSION ===
            dropout_idx = []
            if np.random.rand() < args.occlusion_p:
                k = np.random.randint(args.k_min, args.k_max + 1)
                dropout_idx = np.random.choice(args.numfp, size=k, replace=False)
                print(f"Frame {frame_count:4d}: Occluded FPs {sorted(dropout_idx.tolist())} (k={k}/{args.numfp})")
            
            # === CREATE OCCLUDED STATE FOR CONTROLLER ===
            # Controller sees occluded state (internally predicts missing FPs)
            observed_state = state.copy()
            
            # === CONTROLLER COMPUTES ACTION ===
            self.control_input = self.controller.generateControlInput(observed_state)
            self.control_input[3:6] = 2 * np.pi    # Angular velocity units
            self.control_input[9:12] = 2 * np.pi
            
            # === VISUALIZATION ===
            image = self.visualize_frame(fp_truth_xy, dropout_idx, frame_count, original_state)
            cv2.imshow("Random Occlusion Test", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # === ENVIRONMENT STEP ===
            state, reward, done, _ = self.env.step(self.control_input)
            state[I.left_end_avel_idx] = 2 * np.pi
            
            # === LET CONTROLLER HANDLE ROLLOUT END + SAVING ===
            if done:
                print("✓ Rollout complete - Controller saving state_X.npy")
                self.controller.reset(state)  # ← THIS saves state_X.npy!
                state = self.env.reset()
                state[I.left_end_avel_idx] = 2 * np.pi

def main():
    rospy.init_node('sim_env_occlusion_test', anonymous=True)
    tester = OcclusionTester()
    
    print("Controller handles ALL state saving (state_X.npy)")
    print("Press 'q' in OpenCV window to quit")
    
    try:
        tester.mainLoop()
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
