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
from detect_utils import FpDetector
import json
import print_utils

def load_config(filename):
    with open(filename, 'r') as file:
        config = json.load(file)
    return config


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
        
        
            for key in match_final:
                if match_final[key]!=(0,0): #find matched detection
                    keypoints_vector[key] = match_final[key]
                else: #used previous fp pos to lable
                    miss_idx.append(key)
                    # use last fp position
                    keypoints_vector[key] = keypoints_matrix[-1][key]
        
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
    


if __name__ == "__main__":
    import sys
    import numpy as np
    import pyrealsense2 as rs
    
    config_path = r"C:\Users\91990\Documents\GitHub\FYP_Object_Detection_Model\shape_control_DLO_2\ws_dlo\src\dlo_system_pkg\config\config_tf.json"
    
    detector = dloDetector(config_path)
    print("DLO Detector + RealSense ready! Press 'q' to quit.")
    
    # RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Uncomment for depth
    
    profile = pipeline.start(config)
    
    try:
        while True:
            frameset = pipeline.wait_for_frames()
            color_frame = frameset.get_color_frame()
            if color_frame:
                img = np.asanyarray(color_frame.get_data())
                
                # DLO detection
                # out = detector.showDetectionRegion(img)
                # out2, keypoints, _, _ = detector.detectFeatures(img)  # Uncomment for keypoints
                keypoints_matrix = []
                out, keypoints_3d, keypoints_matrix, misses = detector.detectFeatures(img, keypoints_matrix)
                cv2.imshow("RealSense DLO Detection", out)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Stopped.")


