#!/usr/bin/env python3
"""
Verify tf_img_rb2_2D_onearm:
Click point in camera → script predicts robot X,Y (mm) → move robot there → compare pendant reading
"""
import cv2
import numpy as np
import pyrealsense2 as rs
import json

CONFIG_PATH = "C:/Users/91990/Documents/GitHub/FYP_Object_Detection_Model/shape_control_DLO_2/ws_dlo/src/dlo_system_pkg/config/config_tf.json"

# Load tf matrix
with open(CONFIG_PATH) as f:
    cfg = json.load(f)
tf = np.array(cfg["tf_img_rb2_2D_onearm"])

clicked = []

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked.clear()
        clicked.append((x, y))
        # Predict robot position
        point = np.array([x, y, 0, 1], dtype=float)
        result = tf @ point
        rx_m, ry_m = result[0], result[1]
        rx_mm, ry_mm = rx_m * 1000, ry_m * 1000
        print(f"\n📍 Clicked pixel: ({x}, {y})")
        print(f"🤖 Predicted robot position:")
        print(f"   X = {rx_mm:.1f} mm")
        print(f"   Y = {ry_mm:.1f} mm")
        print(f"   → Move robot TCP to this point on pendant, compare reading!")

# Start camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

cv2.namedWindow("Click to Verify (q to quit)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Click to Verify (q to quit)", 1280, 960)
cv2.setMouseCallback("Click to Verify (q to quit)", mouse_cb)

print("=== TRANSFORMATION MATRIX VERIFIER ===")
print("Click any point on table → compare predicted vs pendant reading")
print("'q' to quit\n")

while True:
    frame = pipeline.wait_for_frames().get_color_frame()
    img = np.asanyarray(frame.get_data())
    vis = img.copy()

    # Draw clicked point
    if clicked:
        x, y = clicked[0]
        cv2.circle(vis, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(vis, f"({x},{y})", (x+10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show prediction on image
        point = np.array([x, y, 0, 1], dtype=float)
        result = tf @ point
        rx_mm, ry_mm = result[0]*1000, result[1]*1000
        cv2.putText(vis, f"X={rx_mm:.1f}mm Y={ry_mm:.1f}mm",
                    (x+10, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(vis, "LEFT-CLICK any point to predict robot position",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("Click to Verify (q to quit)", vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
