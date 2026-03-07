#!/usr/bin/env python3
"""Step 1: Click 4 marked points → record pixel coordinates"""
import cv2
import numpy as np
import pyrealsense2 as rs

pixels = []

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(pixels) < 4:
        pixels.append([float(x), float(y)])
        print(f"Point {len(pixels)}: pixel=({x}, {y})")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

cv2.namedWindow("Click 4 Marked Points")
cv2.setMouseCallback("Click 4 Marked Points", mouse_cb)

print("Click your 4 marked table points in order. 'q' to finish.")

while len(pixels) < 4:
    frame = pipeline.wait_for_frames().get_color_frame()
    img = np.asanyarray(frame.get_data())
    vis = img.copy()
    for i, (u,v) in enumerate(pixels):
        cv2.circle(vis, (int(u),int(v)), 8, (0,255,0), -1)
        cv2.putText(vis, str(i+1), (int(u)+10, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(vis, f"Click point {len(pixels)+1}/4", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("Click 4 Marked Points", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
print("\n📋 COPY THESE PIXELS:")
for i, p in enumerate(pixels):
    print(f"  Point {i+1}: u={p[0]:.0f}, v={p[1]:.0f}")
