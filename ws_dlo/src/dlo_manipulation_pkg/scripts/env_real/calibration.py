#!/usr/bin/env python3
"""
2D camera-to-robot calibration for a fixed RealSense view.

What this script does:
1. Opens RealSense color stream.
2. Lets user click 4 table points in order.
3. Asks user to type robot X,Y (in mm) for each clicked point, in the same order.
4. Computes a 2D affine transform from image pixels -> robot XY (metres).
5. Reports reprojection error.
6. Optionally saves the matrix into config_tf.json.

Assumptions:
- All 4 points lie on the same flat table plane.
- Camera is fixed during and after calibration.
- Robot X,Y values are taken in the same robot base frame you want to use later.
- Z is ignored here and set to 0 in the transform.
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import json
import os

CONFIG_PATH = r"C:\Users\91990\Documents\GitHub\FYP_Object_Detection_Model\shape_control_DLO_2\ws_dlo\src\dlo_system_pkg\config\config_tf.json"
SAVE_KEY = "tf_img_rb2_2D_onearm"
WINDOW_NAME = "2D Calibration - Click 4 Points"

pixels = []


def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(pixels) < 4:
        pixels.append([float(x), float(y)])
        print(f"Point {len(pixels)} pixel = ({x}, {y})")


def ask_robot_points(n=4):
    robot_pts_mm = []
    print("\nNow enter robot TCP X,Y for each clicked point, in the same order.")
    print("Use the values from the pendant/local teach interface.")
    print("Enter in millimetres, example: -423 718\n")

    for i in range(n):
        while True:
            raw = input(f"Point {i+1} robot X Y (mm): ").strip().replace(",", " ")
            parts = raw.split()
            if len(parts) != 2:
                print("Please enter exactly 2 numbers: X Y")
                continue
            try:
                x_mm = float(parts[0])
                y_mm = float(parts[1])
                robot_pts_mm.append([x_mm, y_mm])
                break
            except ValueError:
                print("Invalid number, try again.")
    return np.array(robot_pts_mm, dtype=float)


def compute_affine_tf(pixel_uvs, robot_xys_mm):
    robot_xys = robot_xys_mm / 1000.0

    n = len(pixel_uvs)
    px_mean = pixel_uvs.mean(axis=0)
    rb_mean = robot_xys.mean(axis=0)

    uv_n = pixel_uvs - px_mean
    xy_n = robot_xys - rb_mean

    A = np.hstack([uv_n, np.ones((n, 1))])

    params_x, *_ = np.linalg.lstsq(A, xy_n[:, 0], rcond=None)
    params_y, *_ = np.linalg.lstsq(A, xy_n[:, 1], rcond=None)

    a, b, tx_n = params_x
    c, d, ty_n = params_y

    tx = tx_n - a * px_mean[0] - b * px_mean[1] + rb_mean[0]
    ty = ty_n - c * px_mean[0] - d * px_mean[1] + rb_mean[1]

    tf = np.array([
        [a, b, 0.0, tx],
        [c, d, 0.0, ty],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=float)

    uv_hom = np.c_[pixel_uvs, np.zeros(n), np.ones(n)]
    pred_xy = (tf @ uv_hom.T).T[:, :2]
    errors_mm = np.linalg.norm(pred_xy - robot_xys, axis=1) * 1000.0

    return tf, pred_xy, robot_xys, errors_mm


def save_tf_to_config(tf, config_path, save_key):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
    else:
        cfg = {}

    cfg[save_key] = tf.tolist()

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nSaved matrix to {config_path}")
    print(f"Updated key: {save_key}")


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_cb)

    print("=== 2D CAMERA-ROBOT CALIBRATION ===")
    print("Click 4 table points in order.")
    print("Recommended: use points spread widely across the working area.")
    print("Press 'r' to reset clicks, 'q' to quit.\n")

    try:
        while True:
            frame = pipeline.wait_for_frames().get_color_frame()
            if not frame:
                continue

            img = np.asanyarray(frame.get_data())
            vis = img.copy()

            for i, (u, v) in enumerate(pixels):
                cv2.circle(vis, (int(u), int(v)), 8, (0, 255, 0), -1)
                cv2.putText(vis, f"{i+1}", (int(u) + 10, int(v) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            msg = f"Click point {len(pixels)+1}/4" if len(pixels) < 4 else "4 points selected"
            cv2.putText(vis, msg, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            cv2.imshow(WINDOW_NAME, vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                pixels.clear()
                print("\nClicks reset.\n")
            elif key == ord('q'):
                print("Quit without calibration.")
                return

            if len(pixels) == 4:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    pixel_uvs = np.array(pixels, dtype=float)

    print("\nClicked pixel points:")
    for i, (u, v) in enumerate(pixel_uvs):
        print(f"  Point {i+1}: u={u:.1f}, v={v:.1f}")

    robot_xys_mm = ask_robot_points(n=4)

    tf, pred_xy, robot_xys, errors_mm = compute_affine_tf(pixel_uvs, robot_xys_mm)

    print("\n=== Calibration Result ===")
    print("tf_img_rb2_2D_onearm =")
    print(np.round(tf, 8))

    print("\nPer-point check:")
    for i in range(4):
        pred_mm = pred_xy[i] * 1000.0
        gt_mm = robot_xys[i] * 1000.0
        print(
            f"Point {i+1}: "
            f"pixel=({pixel_uvs[i,0]:.1f}, {pixel_uvs[i,1]:.1f}) | "
            f"robot_gt=({gt_mm[0]:.1f}, {gt_mm[1]:.1f}) mm | "
            f"robot_pred=({pred_mm[0]:.1f}, {pred_mm[1]:.1f}) mm | "
            f"err={errors_mm[i]:.2f} mm"
        )

    print(f"\nMean error: {errors_mm.mean():.2f} mm")
    print(f"Max  error: {errors_mm.max():.2f} mm")

    save = input(f"\nSave matrix to config file key '{SAVE_KEY}'? [y/n]: ").strip().lower()
    if save == "y":
        save_tf_to_config(tf, CONFIG_PATH, SAVE_KEY)
    else:
        print("Matrix not saved.")

    print("\nDone.")
    print("You can now test the matrix by clicking arbitrary points and comparing predicted robot XY.")


if __name__ == "__main__":
    main()
