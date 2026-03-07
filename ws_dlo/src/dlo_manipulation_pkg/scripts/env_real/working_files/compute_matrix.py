#!/usr/bin/env python3
"""Step 3: Compute tf_img_rb2_2D from manual pixel+robot pairs"""
import numpy as np
import json

# ✏️ FILL IN YOUR VALUES:
pixel_uvs = np.array([
    [397, 202],   # Point 1
    [291, 323],   # Point 2
    [391, 413],   # Point 3
    [516, 314],   # Point 4
], dtype=float)

robot_xys_mm = np.array([
    [-423, 718],   # Point 1 (metres)
    [-536, 559],   # Point 2
    [-402, 462],   # Point 3
    [-263, 599],   # Point 4
], dtype=float)

robot_xys = robot_xys_mm / 1000.0  # ← Convert mm → metres here

# Compute affine transform
N = len(pixel_uvs)
px_mean = pixel_uvs.mean(0)
robot_mean = robot_xys.mean(0)
uv_n = pixel_uvs - px_mean
xy_n = robot_xys - robot_mean

# Full affine: robot_x = a*u + b*v + tx
#              robot_y = c*u + d*v + ty
A = np.hstack([uv_n, np.ones((N,1))])   # (4,3)

params_x, *_ = np.linalg.lstsq(A, xy_n[:,0], rcond=None)
params_y, *_ = np.linalg.lstsq(A, xy_n[:,1], rcond=None)
a, b, tx_n = params_x
c, d, ty_n = params_y

tx = tx_n - a*px_mean[0] - b*px_mean[1] + robot_mean[0]
ty = ty_n - c*px_mean[0] - d*px_mean[1] + robot_mean[1]

tf = np.array([
    [a, b, 0.0, tx],
    [c, d, 0.0, ty],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

# Verify error
uv_hom = np.c_[pixel_uvs, np.zeros(N), np.ones(N)]
pred_xy = (tf @ uv_hom.T).T[:, :2]
errors_mm = np.linalg.norm(pred_xy - robot_xys, axis=1) * 1000
print(f"✅ tf_img_rb2_2D:\n{np.round(tf, 8)}")
print(f"Mean error: {errors_mm.mean():.2f}mm  Max: {errors_mm.max():.2f}mm")

# Save
with open("config_tf.json", "r") as f:
    cfg = json.load(f)
cfg["tf_img_rb2_2D"] = tf.tolist()
with open("config_tf.json", "w") as f:
    json.dump(cfg, f, indent=2)
print("Saved to config_tf.json!")
