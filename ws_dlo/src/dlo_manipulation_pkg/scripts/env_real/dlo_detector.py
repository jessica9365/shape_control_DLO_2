#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import necessary libraries
from scipy.interpolate import interp1d
import cv2
import os
import csv
import numpy as np
import glob
import re
from utils.detect_utils import FpDetector
import json
import utils.print_utils as print_utils

def load_config(filename):
    with open(filename, 'r') as file:
        config = json.load(file)
    return config

def interpolateMissingFP(missing_idx, keypoints_vector, last_keypoints, num_fp):
    """
    Estimate position of a missing FP using nearest visible neighbours.
    Falls back to last known position if no neighbours are available.
    """
    # Find nearest visible neighbour to the LEFT
    left_idx, left_pos = None, None
    for i in range(missing_idx - 1, -1, -1):
        if keypoints_vector[i] != (0, 0):
            left_idx = i
            left_pos = keypoints_vector[i]
            break

    # Find nearest visible neighbour to the RIGHT
    right_idx, right_pos = None, None
    for i in range(missing_idx + 1, num_fp):
        if keypoints_vector[i] != (0, 0):
            right_idx = i
            right_pos = keypoints_vector[i]
            break

    # Both neighbours found → linear interpolation
    if left_pos is not None and right_pos is not None:
        # How far is missing_idx between left and right?
        t = (missing_idx - left_idx) / (right_idx - left_idx)
        x = left_pos[0] + t * (right_pos[0] - left_pos[0])
        y = left_pos[1] + t * (right_pos[1] - left_pos[1])
        return (x, y)

    # Only left neighbour → extrapolate using last known velocity
    elif left_pos is not None:
        return last_keypoints[missing_idx]  # fallback to frozen

    # Only right neighbour → fallback
    elif right_pos is not None:
        return last_keypoints[missing_idx]  # fallback to frozen

    # No neighbours at all → frozen
    else:
        return last_keypoints[missing_idx]

class dloDetector():
    def __init__(self, config_path):
        config_path = r"C:\Users\91990\Documents\GitHub\FYP_Object_Detection_Model\shape_control_DLO_2\ws_dlo\src\dlo_system_pkg\config\config_tf.json"
        self.config_path = config_path
        config = load_config(self.config_path)
        print_utils.loginfo(f"Configuration loaded from {config_path}")
        print_utils.loginfo(config)
        self.lower = np.array(config["HSV_lower_range"])
        self.upper = np.array(config["HSV_upper_range"])
        self.pixel_range = config['pixel_range']
        self.num_fp = config["num_fp"]
        self.detector = FpDetector(self.lower, self.upper, self.pixel_range, self.num_fp)
        self.dist_thre = 20 #to trace the keypoints
        
    def showDetectionRegion(self, img):
        return self.detector.drawDetectionRegion(img)
    
    def detectFeatures(self, img, keypoints_matrix=[], dim=2, labels=None, nolabel=False):        
        # Keypoint Alogorithms
        keypoints_vector = [(0, 0)] * self.num_fp
        img_with_color_contour, filtered_contours_by_area = self.detector.colorContour(img)       
        
        # two fps cannot match to one detection, each detection can only be used once
        # corelate detections to fps
        matched_fp_idx = [] # fp index that are already matched, avoid matching one detection to two fp points
        match_detect= {} # key is detection idx
        match_fp = {} # key is fp idx
        match_final = {} #key is fp idx and values are the new fp coordinates
        
        for i in range(self.num_fp):
            match_detect[i] = []
            match_fp[i] = []
            match_final[i] = (0,0)
        
        miss_idx = []
        if keypoints_matrix:
            # track fps and visualize
            for i, cnt_index in enumerate(keypoints_matrix[-1]):  # check keypoints for last frame
                prev_coordinates = cnt_index
        
                find_short = False
                for k, cnt in enumerate(filtered_contours_by_area):  # loop over detected kepoints, find the correponding points in the last frame, detections not in order
        
                    x, y, w, h = cv2.boundingRect(cnt)
                    coordinates = (x + self.detector.startX + w / 2, y + self.detector.startY +h / 2) 
        
                    ecludian_dist = ((coordinates[0] - prev_coordinates[0]) ** 2 + (
                                coordinates[1] - prev_coordinates[1]) ** 2) ** 0.5
        
                    if ecludian_dist < self.dist_thre: # possible match with detections and keypoints
                        # print_utils.loginfo("match", k, i)
        
                        match_detect[k].append((i,ecludian_dist,coordinates))
                        match_fp[i].append((k,ecludian_dist,coordinates))
        
                        match_coordinates = coordinates
                        short_dist = ecludian_dist  # serach for the point with smallest distance
                        find_short = True
        
            for key in match_fp: #sort by the distance, for each fp, what are the nearest detections
                if match_fp[key]:
                    match_fp[key] = sorted(match_fp[key], key=lambda x: x[1])
        
        
            for key in match_detect:
                if match_detect[key]:
                    match_detect[key] = sorted(match_detect[key], key=lambda x: x[1]) # for each detection, find the nearest keypoints
                    for i in range(len(match_detect[key])): # try to match the detection with its near keypoint
                        fp_idx, fp_coords = match_detect[key][i][0], match_detect[key][i][2]
                        if (fp_idx not in matched_fp_idx) and (match_fp[fp_idx][0][0]==key): #if the keypoints has not find the matched detection, and the detection is the nearest point to the fp, match them
                            match_final[fp_idx] = fp_coords
                            matched_fp_idx.append(fp_idx)
                            break
        
        
            # for key in match_final:
            #     if match_final[key]!=(0,0): #find matched detection
            #         keypoints_vector[key] = match_final[key]
            #     else: #used previous fp pos to lable
            #         miss_idx.append(key)
            #         # # use last fp position
            #         # keypoints_vector[key] = keypoints_matrix[-1][key]
            #         # ── NEW: try linear interpolation from neighbours ──
            #         interpolated = interpolateMissingFP(key, keypoints_vector, keypoints_matrix[-1], self.num_fp)
            #         keypoints_vector[key] = interpolated

            # ── REPLACE WITH THIS ────────────────────────────────────────────────────
            # Pass 1: fill all visible (matched) FPs first
            for key in match_final:
                if match_final[key] != (0, 0):
                    keypoints_vector[key] = match_final[key]

            # Pass 2: now that all visible neighbours are populated, interpolate missing FPs
            for key in match_final:
                if match_final[key] == (0, 0):
                    miss_idx.append(key)
                    interpolated = interpolateMissingFP(key, keypoints_vector, keypoints_matrix[-1], self.num_fp)
                    keypoints_vector[key] = interpolated

            print_utils.loginfo(f"cannot find matched for {miss_idx}")
            

        
        
        # initialise keypoint indexes using the first frame in the video
        else:
            if len(self.detector.assignKeypoints(filtered_contours_by_area)) == self.detector.num_fp:  # avoid keypoints not detected in the first few frames
                keypoints_vector = self.detector.assignKeypoints(filtered_contours_by_area)
                print_utils.loginfo(f"Detect all fps : {self.num_fp}")
            else:
                print_utils.logwarn(f"Unable to detect all fp")
        
        # add z dimension to keypoint vector
        keypoints_vector_3D = []
        for keypoint in keypoints_vector:
            keypoints_vector_3D.append((keypoint[0], keypoint[1], 0.0))

        
        keypoints_matrix.append(keypoints_vector)

        if nolabel:
            img_label_fp = img.copy()
        else:
            # img_label_fp = self.detector.label_keypoints_miss(img, keypoints_vector,miss_idx, labels)
            img_label_fp = self.detector.label_keypoints_miss(img, keypoints_vector,miss_idx)

        #-------------------------------------------- pixel value to 3D pos -------------------------------------#
        if dim == 3:
            pass # to fill with actions to get depth

        print_utils.loginfo(keypoints_vector_3D) 
        return img_label_fp, keypoints_vector_3D, keypoints_matrix, miss_idx
    

def main():
    """
    Standalone RealSense test for dlo_detector.py.
    Press 'q' to quit, 'r' to reset keypoints.
    """
    import pyrealsense2 as rs

    config_path = r"C:\Users\91990\Documents\GitHub\FYP_Object_Detection_Model\shape_control_DLO_2\ws_dlo\src\dlo_system_pkg\config\config_tf.json"

    print_utils.loginfo("Initialising dloDetector...")
    dlo = dloDetector(config_path)

    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline_started = False

    keypoints_matrix = []
    frames_to_skip = 10
    first_instance = True  # ← guard flag

    try:
        print_utils.loginfo("Starting RealSense camera...")
        pipeline.start(rs_config)
        pipeline_started = True
        print_utils.loginfo("Camera started. Press 'q' to quit, 'r' to reset.")

        while True:
            frame = pipeline.wait_for_frames()
            color_frame = frame.get_color_frame()
            if not color_frame:
                print_utils.logwarn("No color frame received, skipping...")
                continue

            img = np.asanyarray(color_frame.get_data())

            # Skip first few frames to let auto-exposure settle
            if frames_to_skip > 0:
                frames_to_skip -= 1
                cv2.putText(img, "Loading... please wait.", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.imshow("dlo_detector test", img)
                cv2.waitKey(1)
                continue

            # Run detection
            img_out, keypoints_3d, keypoints_matrix, miss_idx = dlo.detectFeatures(
                img, keypoints_matrix
            )

            # ── First instance guard ──────────────────────────────────────────
            if first_instance:
                if not any((x, y) != (0.0, 0.0) for x, y, z in keypoints_3d):
                    print_utils.logwarn("No keypoints detected on first frame, retrying...")
                    keypoints_matrix = []  # discard bad initialisation row
                else:
                    print_utils.loginfo(f"Keypoints initialised successfully.")
                    first_instance = False
                cv2.imshow("dlo_detector test", img_out)
                cv2.waitKey(1)
                continue  # don't proceed to tracking until initialisation is clean
            # ─────────────────────────────────────────────────────────────────

            # Show detection region box
            img_out = dlo.showDetectionRegion(img_out)

            # Overlay FP index labels
            for i, (x, y, z) in enumerate(keypoints_3d):
                if (x, y) != (0.0, 0.0):
                    cv2.putText(img_out, str(i), (int(x), int(y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # Status text
            detected = sum(1 for x, y, z in keypoints_3d if (x, y) != (0.0, 0.0))
            cv2.putText(img_out, f"FPs: {detected}/{dlo.num_fp}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if miss_idx:
                cv2.putText(img_out, f"Missing: {miss_idx}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow("dlo_detector test", img_out)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print_utils.loginfo("Quit.")
                break
            elif key == ord('r'):
                print_utils.loginfo("Resetting keypoints...")
                keypoints_matrix = []
                first_instance = True  # ← also reset the guard on manual reset

    except Exception as e:
        import sys
        tb = sys.exc_info()[2]
        print_utils.logerr(f"LINE {tb.tb_lineno}: {e}")

    finally:
        if pipeline_started:
            pipeline.stop()
        cv2.destroyAllWindows()
        print_utils.loginfo("Test complete.")


if __name__ == "__main__":
    main()
