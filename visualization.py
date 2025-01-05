import numpy as np
import math
import rerun as rr
from rerun import RotationAxisAngle, Angle
from constants import TAG_SIZE, DASH_MOVEMENT_CONSTRAINT, ROBOT_HEIGHT
from utils import axis_angle_from_matrix

class RobotVisualizer:
    def __init__(self):
        # Initialize rerun
        rr.init("my_swerve_rerun", spawn=True)

    def update_2d(self, latest_odom, latest_modules):
        """
        Re-logs the position of the robot and the direction of each module
        in a single 2D view, using SE2 transforms to place everything.
        """
        x = latest_odom["x"]
        y = -latest_odom["y"]
        theta = -latest_odom["theta"]  # in radians

        # Grab module angles:
        fl_angle = latest_modules["front_left"]
        fr_angle = latest_modules["front_right"]
        bl_angle = latest_modules["back_left"]
        br_angle = latest_modules["back_right"]

        # Robot corners in local frame
        half = DASH_MOVEMENT_CONSTRAINT.robot_radius
        local_corners = [
            ( half,  half),
            ( half, -half),
            (-half, -half),
            (-half,  half),
            ( half,  half),
        ]

        # Transform corners to global frame
        global_corners = []
        for (cx, cy) in local_corners:
            gx, gy = self.to_global(cx, cy, x, y, theta)
            global_corners.append((gx, gy))

        # Log robot outline
        rr.log(
            "robot_outline",
            rr.LineStrips2D(
                [global_corners],
                colors=[[128, 128, 128]],
                radii=0.002,
            ),
        )

        # Log heading arrow
        heading_length = half
        arrow_end_x = x + heading_length * math.cos(theta)
        arrow_end_y = y + heading_length * math.sin(theta)

        rr.log(
            "robot_heading",
            rr.Arrows2D(
                origins=[[x, y]],
                vectors=[[arrow_end_x - x, arrow_end_y - y]],
                colors=[[255, 0, 0]],
                show_labels=[False],
                radii=0.003,
            ),
        )

        # Log module arrows
        module_arrow_len = 0.05
        self._log_module_arrow(x, y, theta, half, half, fl_angle, "front_left_module", module_arrow_len)
        self._log_module_arrow(x, y, theta, half, -half, fr_angle, "front_right_module", module_arrow_len)
        self._log_module_arrow(x, y, theta, -half, half, bl_angle, "back_left_module", module_arrow_len)
        self._log_module_arrow(x, y, theta, -half, -half, br_angle, "back_right_module", module_arrow_len)

    def update_3d(self, latest_odom, latest_modules):
        """
        Log a 3D bounding box for the robot, a pinned transform for a camera,
        and 3D arrows for each swerve module's orientation.
        """
        x = latest_odom["x"]
        y = latest_odom["y"]
        theta = latest_odom["theta"] - np.pi / 2.0

        # Robot dimensions
        half_x = DASH_MOVEMENT_CONSTRAINT.robot_radius
        half_y = DASH_MOVEMENT_CONSTRAINT.robot_radius
        half_z = ROBOT_HEIGHT / 2.0

        # Robot base transform
        half_angle = theta / 2.0
        s = math.sin(half_angle)
        c = math.cos(half_angle)

        rr.log(
            "robot/base",
            rr.Boxes3D(
                centers=[[x, y, half_z]],
                half_sizes=[[half_x, half_y, half_z]],
                quaternions=[[0.0, 0.0, s, c]],
                colors=[[0, 255, 255, 128]],
                fill_mode="solid",
            ),
        )

        # Camera transform
        pitch_deg = 0.0
        pitch_rad = math.radians(pitch_deg)
        roll_deg = -90.0
        roll_rad = math.radians(roll_deg)

        # Build rotation matrices
        pitch_mat = np.array([
            [ math.cos(pitch_rad), 0.0, math.sin(pitch_rad)],
            [ 0.0,                1.0,              0.0     ],
            [-math.sin(pitch_rad), 0.0, math.cos(pitch_rad)],
        ], dtype=float)

        yaw_mat = np.array([
            [ math.cos(theta), -math.sin(theta), 0.0],
            [ math.sin(theta),  math.cos(theta), 0.0],
            [ 0.0,             0.0,             1.0],
        ], dtype=float)

        roll_mat = np.array([
            [1.0,         0.0,          0.0],
            [0.0,  math.cos(roll_rad), -math.sin(roll_rad)],
            [0.0,  math.sin(roll_rad),  math.cos(roll_rad)],
        ], dtype=float)

        final_rot = yaw_mat @ pitch_mat @ roll_mat
        axis, rot_angle = axis_angle_from_matrix(final_rot)

        rr.log(
            "robot/camera",
            rr.Transform3D(
                translation=[x, y, ROBOT_HEIGHT],
                rotation=RotationAxisAngle(axis=axis, angle=Angle(rad=rot_angle)),
            ),
        )

        # Log camera pinhole
        fx, fy = 597.19, 598.65
        w, h = 1920, 1080
        rr.log("robot/camera", rr.Pinhole(focal_length=(fx, fy), width=w, height=h, image_plane_distance=1.0))

        # Log module arrows
        fl_angle = latest_modules["front_left"]
        fr_angle = latest_modules["front_right"]
        bl_angle = latest_modules["back_left"]
        br_angle = latest_modules["back_right"]

        self._log_module_arrow_3d(x, y, theta, half_x, half_y, fl_angle, "front_left_module")
        self._log_module_arrow_3d(x, y, theta, half_x, -half_y, fr_angle, "front_right_module")
        self._log_module_arrow_3d(x, y, theta, -half_x, half_y, bl_angle, "back_left_module")
        self._log_module_arrow_3d(x, y, theta, -half_x, -half_y, br_angle, "back_right_module")

    def log_apriltag(self, tag_id, tag_in_cam, latest_odom):
        """Log an AprilTag visualization given its pose in camera frame"""
        robot_theta = latest_odom["theta"]
        cx = latest_odom["x"]
        cy = latest_odom["y"]
        # 10 inches above the robot
        cz = 10.0 * 0.0254

        # Build camera transform
        CamGlobal = np.eye(4)
        CamGlobal[0, 3] = cx
        CamGlobal[1, 3] = cy
        CamGlobal[2, 3] = cz
        CamGlobal[0, 0] = np.cos(robot_theta)
        CamGlobal[0, 1] = -np.sin(robot_theta)
        CamGlobal[1, 0] = np.sin(robot_theta)
        CamGlobal[1, 1] = np.cos(robot_theta)

        # Transform tag to global frame
        tag_global = CamGlobal @ tag_in_cam
        tx, ty, tz = tag_global[0, 3], tag_global[1, 3], tag_global[2, 3]
        rotation_mat = tag_global[:3, :3]
        axis, angle = axis_angle_from_matrix(rotation_mat)

        # Log tag visualization
        half_sz = TAG_SIZE / 2.0
        rr.log(
            f"robot/tag_{tag_id}",
            rr.Transform3D(
                translation=[tx, ty, tz],
                rotation=RotationAxisAngle(axis=axis, angle=Angle(rad=angle)),
            ),
        )
        rr.log(
            f"robot/tag_{tag_id}/box",
            rr.Boxes3D(
                centers=[[0, 0, 0]],
                half_sizes=[[half_sz, half_sz, 0.01]],
                colors=[[255, 255, 0]],
                fill_mode="solid",
            ),
        )

    def log_image(self, image):
        """Log a camera image"""
        rr.log("robot/camera", rr.Image(image))

    def log_wheel_velocities(self, velocities):
        """Log wheel velocities as scalars"""
        rr.log("wheel_vels/front_left", rr.Scalar(velocities["front_left"]))
        rr.log("wheel_vels/front_right", rr.Scalar(velocities["front_right"]))
        rr.log("wheel_vels/back_left", rr.Scalar(velocities["back_left"]))
        rr.log("wheel_vels/back_right", rr.Scalar(velocities["back_right"]))

    def log_module_angles(self, angles):
        """Log module angles as scalars"""
        rr.log("module_angles/front_left", rr.Scalar(angles["front_left"]))
        rr.log("module_angles/front_right", rr.Scalar(angles["front_right"]))
        rr.log("module_angles/back_left", rr.Scalar(angles["back_left"]))
        rr.log("module_angles/back_right", rr.Scalar(angles["back_right"]))

    def log_lidar_scan(self, points, latest_odom):
        """Log lidar scan points in global frame"""
        points_global = []
        for p in points:
            gx, gy = self.to_global(
                p["x"],
                -p["y"],
                latest_odom["x"],
                -latest_odom["y"],
                latest_odom["theta"]
            )
            points_global.append([gx, gy])
        rr.log("lidar_scan", rr.Points2D(points_global))

    @staticmethod
    def to_global(px, py, rx, ry, rtheta):
        """Convert local point (px,py) to global given robot pose (rx,ry,rtheta)"""
        cosT = math.cos(rtheta)
        sinT = math.sin(rtheta)
        gx = rx + (px * cosT - py * sinT)
        gy = ry + (px * sinT + py * cosT)
        return gx, gy

    def _log_module_arrow(self, x, y, theta, offset_x, offset_y, module_angle, name, arrow_len):
        """Helper to log a 2D module direction arrow"""
        mod_origin = self.to_global(offset_x, offset_y, x, y, theta)
        total_angle = theta + module_angle
        vx = arrow_len * math.cos(total_angle)
        vy = arrow_len * math.sin(total_angle)

        rr.log(
            name,
            rr.Arrows2D(
                origins=[mod_origin],
                vectors=[[vx, vy]],
                colors=[[0, 255, 0]],
                show_labels=[False],
                radii=0.003,
            ),
        )

    def _log_module_arrow_3d(self, x, y, theta, offset_x, offset_y, module_angle, name):
        """Helper to log a 3D module direction arrow"""
        cosT = math.cos(theta)
        sinT = math.sin(theta)
        gx = x + (offset_x * cosT - offset_y * sinT)
        gy = y + (offset_x * sinT + offset_y * cosT)
        gz = 0.0

        total_angle = theta + module_angle
        arrow_len = 0.05
        vx = arrow_len * math.cos(total_angle)
        vy = arrow_len * math.sin(total_angle)
        vz = 0.0

        rr.log(
            f"robot/modules/{name}",
            rr.Arrows3D(
                origins=[[gx, gy, gz]],
                vectors=[[vx, vy, vz]],
                radii=0.003,
                colors=[[0, 255, 0]],
                show_labels=[False],
            ),
        ) 