#!/usr/bin/env python3
"""
Complete Camera-Robot Calibration for ur_desiredshapegenerator.
Click 4 points after moving TCP there → outputs tf_img_rb1_2D for config_tf.json
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import json
import time
from ur_controller import URrobot  # Same as your script
import print_utils  # Your print_utils.py file

# Robot IPs from your script [file:1]
# ROBOT_IP_LEFT = '10.149.230.2'   # NTU_UR5E_LOAN_IP (left base frame)
# ROBOT_IP_RIGHT = '10.149.230.1'  # NTU_UR5E_MAIN_IP
ROBOT_IP_RIGHT = '169.254.149.80'  # NTU_UR5E_MAIN_IP

# def fit_affine_2d(pixel_uv, robot_xy):
#     """Compute 4x4 tf_img_rb1_2D matrix"""
#     N = len(pixel_uv)

#     A = np.hstack([
#         np.c_[pixel_uv, np.ones((N,1))], 
#         np.c_[np.zeros((N,3)), pixel_uv, np.ones((N,1))]
#     ])

#     b = np.hstack([robot_xy[:,0], robot_xy[:,1]])
#     params = np.linalg.lstsq(A, b, rcond=None)[0]
#     a, b1, tx, c, d, ty = params
#     return np.array([
#         [a, b1, 0.0, tx],
#         [c,  d,  0.0, ty],
#         [0.0,0.0, 1.0, 0.0],
#         [0.0,0.0, 0.0, 1.0]
#     ])

def fit_affine_2d(pixel_uv, robot_xy):
    pixel_uv = np.array(pixel_uv, dtype=float)
    robot_xy = np.array(robot_xy, dtype=float)
    N = len(pixel_uv)
    
    # Normalize
    px_mean = pixel_uv.mean(0)
    rx_mean = robot_xy.mean(0)
    pixel_uv_n = pixel_uv - px_mean
    robot_xy_n = robot_xy - rx_mean
    
    # A: [u, 0, 1, 0, v, 1] for X/Y
    A = np.hstack([
        pixel_uv_n[:,0:1], np.zeros((N,1)), np.ones((N,1)),  # a*u + 0 + tx
        np.zeros((N,1)), pixel_uv_n[:,1:2], np.ones((N,1))   # 0 + d*v + ty
    ])
    
    # Solve X: a*u + tx = robot_x
    params_x, *_ = np.linalg.lstsq(A, robot_xy_n[:,0], rcond=None)
    a, _, tx_n, _, _, _ = params_x
    
    # Solve Y: d*v + ty = robot_y  
    A_y = A[:, [4,5]]  # v, 1 cols only
    params_y, *_ = np.linalg.lstsq(A_y, robot_xy_n[:,1], rcond=None)
    d, ty_n = params_y
    
    # Assume orthogonal (b1=c=0 for 2D scale+translate)
    b1 = c = 0.0
    
    # Denormalize tx/ty
    tx = tx_n - a*px_mean[0]
    ty = ty_n - d*px_mean[1]
    
    tf = np.array([
        [a, b1, 0.0, tx],
        [c, d, 0.0, ty],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    return tf


def main():
    print_utils.loginfo("=== CAMERA-ROBOT CALIBRATION ===")
    
    # Connect robots (same as your main script)
    print_utils.loginfo(f"Connecting right robot {ROBOT_IP_RIGHT}...")
    # left_robot = URrobot(ROBOT_IP_LEFT)
    right_robot = URrobot(ROBOT_IP_RIGHT)
    
    if not (right_robot.is_online):
        print_utils.logerr("ERROR: Enable Remote Control on pendants first!")
        return
    
    robot = right_robot  # Calibrate in RIGHT base frame [file:1]
    print_utils.loginfo("Robot's online. Use pendant LOCAL control to move TCP.")
    
    # RealSense (exact from your script [file:1])
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    
    pixel_uvs = []
    robot_xys = []
    
    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pixel_uvs) < 4:
            pose = robot.receive.getActualTCPPose()  # [x,y,z,rx,ry,rz] [file:1]
            pixel_uvs.append([float(x), float(y)])
            robot_xys.append([pose[0], pose[1]])  # x,y meters
            print_utils.loginfo(f"✅ Point {len(pixel_uvs)}: pixel=({x:.0f},{y:.0f}) robot=({pose[0]:.4f},{pose[1]:.4f})")
    
    cv2.namedWindow("CALIBRATE (Left-Click 4 Points)")
    cv2.setMouseCallback("CALIBRATE (Left-Click 4 Points)", mouse_cb)
    
    print("\n📋 INSTRUCTIONS:")
    print("1. Pendant LOCAL → Move TCP to table point #1")
    print("2. MOUSE LEFT-CLICK exact point in camera image")
    print("3. Repeat for 4 well-spread points")
    print("'q' to quit\n")
    
    while len(pixel_uvs) < 4:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color = aligned.get_color_frame()
        if not color: continue
        
        img = np.asanyarray(color.get_data())
        vis = img.copy()
        
        # Draw previous points
        for i, (u,v) in enumerate(pixel_uvs):
            cv2.circle(vis, (int(u),int(v)), 10, (0,255,0), 4)
            cv2.putText(vis, str(i+1), (int(u)+15, int(v)-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        # Instructions overlay
        cv2.putText(vis, f"Point {len(pixel_uvs)+1}/4 - CLICK!", (20,40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(vis, "Press 'q' to quit", (20,70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        cv2.imshow("CALIBRATE (Left-Click 4 Points)", vis)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
    
    pipeline.stop()
    cv2.destroyAllWindows()
    
    if len(pixel_uvs) < 4:
        print_utils.logerr("Need 4 points!")
        return
    
    # COMPUTE & SAVE
    tf_img_rb1_2D = fit_affine_2d(np.array(pixel_uvs), np.array(robot_xys))
    
    # ERROR CHECK
    uv_hom = np.c_[pixel_uvs, np.zeros((4,1)), np.ones((4,1))]
    pred_xy = (tf_img_rb1_2D @ uv_hom.T).T[:, :2]
    errors_mm = np.linalg.norm(pred_xy - np.array(robot_xys), axis=1) * 1000
    mean_err = errors_mm.mean()
    
    print_utils.loginfo("\n🎉 CALIBRATION COMPLETE!")
    print_utils.loginfo(f"tf_img_rb2_2D =\n{np.round(tf_img_rb1_2D, decimals=8)}")
    print_utils.loginfo(f"Mean error: {mean_err:.2f}mm (good if <3mm)")
    
    # SAVE config_tf.json
    config = {
        "tf_img_rb2_2D": tf_img_rb1_2D.tolist(),
        "pixel_range": [10, 10, 600, 430],
        "HSV_lower_range": [102, 89, 0],
        "HSV_upper_range": [117, 255, 255],
        "num_fp": 10,  # From your config
        "dlo_length": 0.5
    }
    with open("config_tf.json", "w") as f:
        json.dump(config, f, indent=2)
    print_utils.loginfo("SAVED config_tf.json")
    print_utils.loginfo("Copy 'tf_img_rb2_2D' to your existing config_tf.json if needed.")

if __name__ == "__main__":
    main()
