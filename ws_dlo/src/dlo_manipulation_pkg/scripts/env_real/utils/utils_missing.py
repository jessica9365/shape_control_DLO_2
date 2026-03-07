import numpy as np
import cv2
from utils.data_utils import  to_pixels
def create_miss_rectangle (target_pos, fp_pos, width=0.15, height=0.15):
    # Get midpoint between points 4 and 5
    mid_point = (target_pos[3, :2] + target_pos[4, :2]) / 2.0
    # mid_point = (target_pos[3, :2] + target_pos[4, :2]) / 2.0
    # Fixed rectangle size
    rect_width = width
    rect_height = height

    # Compute bounds centered at midpoint
    x0 = mid_point[0] - rect_width / 2
    x1 = mid_point[0] + rect_width / 2
    y0 = mid_point[1] - rect_height / 2
    y1 = mid_point[1] + rect_height / 2

    print(f"Rect bounds: x=[{x0:.3f}, {x1:.3f}], y=[{y0:.3f}, {y1:.3f}]")

    # 3. Check which fp_pos points fall inside the rectangle
    inside_mask = (
            (fp_pos[:, 0] >= x0) & (fp_pos[:, 0] <= x1) &
            (fp_pos[:, 1] >= y0) & (fp_pos[:, 1] <= y1)
    )

    # 4. Get indices of the covered / missing points
    missing_idx = list(np.where(inside_mask)[0])

    return missing_idx, [x0, x1, y0, y1]

def draw_image_w_miss (x_true, y_true, x_miss, y_miss, x_target, y_target, missing_idx, bbox_coords, num_fp=8):
    image = np.full((882, 1568, 3), 255, dtype=np.uint8)
    overlay = image.copy()
    # Draw blue circles and lines (x_true, y_true)
    points_true = []
    for x, y in zip(x_true, y_true):
        pt = (int(x), int(y))
        cv2.circle(image, pt, 10, (255, 0, 0), -1)  # Blue
        points_true.append(pt)
    for i in range(1, len(points_true)):
        cv2.line(image, points_true[i - 1], points_true[i], (255, 0, 0), 2)

    # Blend overlay with original image (alpha = 0.5)
    alpha = 0.4  # 20% transparency
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Draw red circles and lines (x_miss, y_miss)
    # Draw red circles and lines only for points in drop_idx
    try:
        points_miss = []
        new_lst = [i+1 for i in missing_idx]
        for i, (x, y) in enumerate(zip(x_miss, y_miss)):
            # if i not in new_lst:  # skip points not in drop_idx
            #     continue

            if i == 0 or i == num_fp+1:
                radius = 15  # big
            else:
                radius = 10  # normal

            pt = (int(x), int(y))
            cv2.circle(image, pt, radius, (0, 0, 255), -1)  # Red
            points_miss.append(pt)

        # Draw connecting lines only for the drawn points
        for i in range(1, len(points_miss)):
            cv2.line(image, points_miss[i - 1], points_miss[i], (0, 0, 255), 2)

    except Exception as e:
        print("Error drawing points_miss:", e)

    # Draw green circles and lines (target)
    points_target = []
    for x, y in zip(x_target, y_target):
        pt = (int(x), int(y))
        cv2.circle(image, pt, 10, (0, 255, 0), -1)  # Green
        points_target.append(pt)
    for i in range(1, len(points_target)):
        cv2.line(image, points_target[i - 1], points_target[i], (0, 255, 0), 2)

    # Draw green rectangles and connecting lines (target)
    points_target = []
    rect_w, rect_h = 10, 10  # rectangle width and height

    for x, y in zip(x_target, y_target):
        cx, cy = int(x), int(y)

        # Rectangle coordinates (centered at point)
        x1_tar = cx - rect_w // 2
        y1_tar = cy - rect_h // 2
        x2_tar = cx + rect_w // 2
        y2_tar = cy + rect_h // 2

        # cv2.rectangle(image, (x1_tar, y1_tar), (x2_tar, y2_tar), (0, 255, 0), -1)  # filled green rectangle

        radius = 10
        cv2.circle(image, (cx,cy), radius, (0, 255, 0), -1)  # Red

        points_target.append((cx, cy))

    # Draw connecting lines
    for i in range(1, len(points_target)):
        cv2.line(image, points_target[i - 1], points_target[i], (0, 255, 0), 2)

    # Draw the occluded rectangle (green border)
    if bbox_coords:
        x0, x1, y0, y1 = bbox_coords
        x_rect_pix, y_rect_pix = to_pixels(np.array([[x0, y0], [x1, y1]]), 1920,1080, x_range=(-1, 1),y_range=(-1, 1))
        pt1 = tuple(np.array([x_rect_pix[0], y_rect_pix[0]]).astype(int))
        pt2 = tuple(np.array([x_rect_pix[1], y_rect_pix[1]]).astype(int))
        cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)

        # --- Draw shaded rectangle ---
        overlay = image.copy()

        # Filled rectangle on overlay (green-ish)
        shade_color = (0, 255, 0)  # green
        cv2.rectangle(overlay, pt1, pt2, shade_color, -1)

        # Blend with transparency
        alpha = 0.3  # 30% opacity, adjust as needed
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # --- Optional: draw border line on top ---
        cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)

    cv2.putText(image, "Green: Target", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, "Blue: Real State", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(image, "Red: Prediction", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Text to be written on each frame
    # text = f"Rollout {rollout_idx} - Frame {frame}"
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # # Put the text on the frame
    # cv2.putText(image, text, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return image

import numpy as np

def interpolate_missing_points(observed_fp_pos: np.ndarray,
                               missing_idx,
                               inplace: bool = True):
    """
    Linear interpolation over missing indices (supports consecutive runs).

    observed_fp_pos: [N, D] array of positions (will be modified if inplace=True)
                     It should contain valid values for non-missing indices.
    missing_idx: iterable of missing indices (e.g., [2,3,7])
    inplace: if False, returns a new array.

    Behavior:
      - For an interior missing run [a..b], interpolate between left=a-1 (nearest observed)
        and right=b+1 (nearest observed), even if gaps are larger.
      - If missing run touches boundary (no left or no right), it uses nearest observed
        and does constant fill (you can change to extrapolation if you want).
    """
    x = observed_fp_pos if inplace else observed_fp_pos.copy()
    N, D = x.shape

    miss = np.zeros(N, dtype=bool)
    miss[list(missing_idx)] = True

    i = 0
    while i < N:
        if not miss[i]:
            i += 1
            continue

        # find run [start, end]
        start = i
        while i < N and miss[i]:
            i += 1
        end = i - 1

        # find nearest observed left and right indices
        left = start - 1
        while left >= 0 and miss[left]:
            left -= 1

        right = end + 1
        while right < N and miss[right]:
            right += 1

        run_len = end - start + 1

        if left >= 0 and right < N:
            # interpolate between x[left] and x[right]
            L = x[left].copy()
            R = x[right].copy()
            for k in range(run_len):
                alpha = (k + 1) / (run_len + 1)  # 1/(m+1), ..., m/(m+1)
                x[start + k] = (1 - alpha) * L + alpha * R
        elif left >= 0 and right >= N:
            # only left exists: constant fill with left
            x[start:end+1] = x[left]
        elif left < 0 and right < N:
            # only right exists: constant fill with right
            x[start:end+1] = x[right]
        else:
            # all points missing (degenerate)
            raise ValueError("All points are missing; cannot interpolate.")

    return x
