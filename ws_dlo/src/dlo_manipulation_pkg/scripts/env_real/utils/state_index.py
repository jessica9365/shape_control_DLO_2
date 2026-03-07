# utils/state_index.py

try:
    import rospy
except ImportError:
    rospy = None

# If you want to still support ROS param when available:
# num_fps = rospy.get_param("DLO/num_FPs") if rospy else 10
num_fps = 10


class Index(object):
    """
    Backward-compatible indexing for:
      - v1 layout (original): 117 dims when num_fps=10
      - v2 layout: v1 + fp_visible_mask (num_fps dims) appended at the end

    v1 indices are unchanged.
    v2 adds:
      - fp_visible_mask_idx (length num_fps)
      - state_dim_v1, state_dim_v2
    """
    def __init__(self, num_fps: int = 10, with_mask: bool = False):
        self.num_fps = int(num_fps)
        self.with_mask = bool(with_mask)

        # -------------------------
        # v1 layout (UNCHANGED)
        # -------------------------
        length_start = 0
        length_end = 1

        fps_pos_start = length_end
        fps_pos_end = fps_pos_start + 3 * self.num_fps

        end_pose_start = fps_pos_end
        end_pose_end = end_pose_start + 14

        fps_vel_start = end_pose_end
        fps_vel_end = fps_vel_start + 3 * self.num_fps

        ends_vel_start = fps_vel_end
        ends_vel_end = ends_vel_start + 12

        desired_pos_start = ends_vel_end
        desired_pos_end = desired_pos_start + 3 * self.num_fps

        # v1 dimension
        self.state_dim_v1 = desired_pos_end

        # original name kept for backward compatibility
        # (old code expects I.state_dim to mean v1)
        self.state_dim = self.state_dim_v1

        self.length_idx = list(range(length_start, length_end))
        self.fps_pos_idx = list(range(fps_pos_start, fps_pos_end))
        self.end_pose_idx = list(range(end_pose_start, end_pose_end))
        self.fps_vel_idx = list(range(fps_vel_start, fps_vel_end))
        self.ends_vel_idx = list(range(ends_vel_start, ends_vel_end))
        self.desired_pos_idx = list(range(desired_pos_start, desired_pos_end))

        self.left_end_pos_idx = self.end_pose_idx[0:3]
        self.left_end_quat_idx = self.end_pose_idx[3:7]
        self.right_end_pos_idx = self.end_pose_idx[7:10]
        self.right_end_quat_idx = self.end_pose_idx[10:14]

        self.left_end_vel_idx = self.ends_vel_idx[0:6]
        self.right_end_vel_idx = self.ends_vel_idx[6:12]
        self.left_end_lvel_idx = self.ends_vel_idx[0:3]
        self.left_end_avel_idx = self.ends_vel_idx[3:6]
        self.right_end_lvel_idx = self.ends_vel_idx[6:9]
        self.right_end_avel_idx = self.ends_vel_idx[9:12]

        # what your model uses as input (UNCHANGED)
        self.state_input_idx = self.fps_pos_idx + self.end_pose_idx

        # -------------------------
        # v2 additions (mask appended)
        # -------------------------
        mask_start = self.state_dim_v1
        mask_end = mask_start + self.num_fps
        self.fp_visible_mask_idx = list(range(mask_start, mask_end))

        self.state_dim_v2 = mask_end

        # If you want Index(with_mask=True) to represent v2 as "the state"
        if self.with_mask:
            self.state_dim = self.state_dim_v2

    def assert_is_v1(self, arr):
        assert arr.shape[1] == self.state_dim_v1, f"Expected v1 dim {self.state_dim_v1}, got {arr.shape[1]}"

    def assert_is_v2(self, arr):
        assert arr.shape[1] == self.state_dim_v2, f"Expected v2 dim {self.state_dim_v2}, got {arr.shape[1]}"


# Default instance behaves like your old code (v1)
I = Index(num_fps=num_fps, with_mask=False)

# Optional: convenience instance for v2
I2 = Index(num_fps=num_fps, with_mask=True)
