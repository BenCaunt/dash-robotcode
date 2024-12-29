import json
import time
import zenoh
import pygame
import math
import threading
import cv2
import numpy as np

# ---- NEW IMPORTS FOR RERUN:
import rerun as rr
from geometry2d import Transform2d, Vector2d
from auton import autonomous_motion, RobotContext, autonomous_running
from constants import (
    VELOCITY_KEY, ZERO_HEADING_KEY, MEASURED_TWIST_KEY,
    ODOMETRY_KEY, WHEEL_VELOCITIES_KEY, MODULE_ANGLES_KEY,
    LIDAR_SCAN_KEY, DASH_MOVEMENT_CONSTRAINT, TAG_SIZE
)
from rerun import RotationAxisAngle, Angle

from utils import angle_wrap, axis_angle_from_matrix

LEFT_X_AXIS = 0
LEFT_Y_AXIS = 1
RIGHT_X_AXIS = 2
CIRCLE_BUTTON_INDEX = 1
CROSS_BUTTON_INDEX = 0  # For example, set to whichever index CROSS should be

# Robot dimensions in meters (10 inch = 0.254 m).
# We'll model the robot as a square for simplicity:
ROBOT_SIZE = 0.254
ROBOT_HEIGHT = 0.254  # new - 10 inch height

# Retrieve control scaling from constraints
max_speed = DASH_MOVEMENT_CONSTRAINT.max_velocity.vx
max_omega_deg = DASH_MOVEMENT_CONSTRAINT.max_velocity.w

# Track global states for visualization:
latest_odom = {"x": 0.0, "y": 0.0, "theta": 0.0}
latest_modules = {
    "front_left": 0.0,
    "front_right": 0.0,
    "back_left": 0.0,
    "back_right": 0.0,
}


def apply_deadband(value, deadband=0.05):
    if abs(value) < deadband:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    scaled = (abs(value) - deadband) / (1 - deadband)
    return sign * scaled

def normalize_axis(value):
    return value

def wheel_velocities_callback(sample):
    data = json.loads(sample.payload.to_string())
    front_left = data["front_left"]
    front_right = data["front_right"]
    back_left = data["back_left"]
    back_right = data["back_right"]

    rr.log("wheel_vels/front_left", rr.Scalar(front_left))
    rr.log("wheel_vels/front_right", rr.Scalar(front_right))
    rr.log("wheel_vels/back_left", rr.Scalar(back_left))
    rr.log("wheel_vels/back_right", rr.Scalar(back_right))

def module_angles_callback(sample):
    """
    Store the module angles (in radians) and log a visualization update.
    """
    global latest_modules
    data = json.loads(sample.payload.to_string())

    latest_modules["front_left"]  = data["front_left"]
    latest_modules["front_right"] = data["front_right"]
    latest_modules["back_left"]   = data["back_left"]
    latest_modules["back_right"]  = data["back_right"]

    rr.log("module_angles/front_left", rr.Scalar(latest_modules["front_left"]))
    rr.log("module_angles/front_right", rr.Scalar(latest_modules["front_right"]))
    rr.log("module_angles/back_left", rr.Scalar(latest_modules["back_left"]))
    rr.log("module_angles/back_right", rr.Scalar(latest_modules["back_right"]))

    # Log new visualization since modules changed:
    update_rerun_viz()

def odom_listener(sample):
    """
    Update the global odometry and log a visualization update.
    """
    global latest_odom
    data_str = sample.payload.to_string()
    try:
        data = json.loads(data_str)
        latest_odom["x"] = data["x"]
        latest_odom["y"] = data["y"]
        latest_odom["theta"] = data["theta"]  # presumably in radians
        # Log new visualization since odom changed:
        update_rerun_viz()
    except Exception as e:
        print(f"Failed to parse odom: {e}")

def measured_twist_listener(sample):
    data_str = sample.payload.to_string()
    try:
        data = json.loads(data_str)
        vx = data["vx"]
        vy = data["vy"]
        omega = data["omega"]  # rad/s
    except Exception as e:
        print(f"Failed to parse measured twist: {e}")

def to_global(px, py, rx, ry, rtheta):
    """
    Use geometry2d.Transform2d to convert local (px, py) into global coords,
    given robot pose (rx, ry, rtheta).
    """
    # Create a Transform2d for the robot’s global pose:
    robot_tf = Transform2d(rx, ry, rtheta)
    # Create a Transform2d for the point in local frame:
    point_local_tf = Transform2d(px, py, 0.0)
    # Combine them:
    point_global_tf = robot_tf * point_local_tf
    return (point_global_tf.x, point_global_tf.y)

def lidar_callback(sample):
    data_str = sample.payload.to_string()
    try:
        data = json.loads(data_str)  # expecting a list of { "x": float, "y": float }
        points_global = []
        for p in data:
            gx, gy = to_global(
                p["x"],
                -p["y"],  # Mirror Y usage
                latest_odom["x"],
                -latest_odom["y"],
                latest_odom["theta"]
            )
            points_global.append([gx, gy])
        rr.log("lidar_scan", rr.Points2D(points_global))
    except Exception as e:
        print(f"Failed to parse lidar scan: {e}")

def update_rerun_viz():
    """
    Re-logs the position of the robot and the direction of each module
    in a single 2D view, using SE2 transforms to place everything.
    """

    # Grab the current pose:
    x = latest_odom["x"]
    y = -latest_odom["y"]
    theta = -latest_odom["theta"]  # in radians

    # Grab module angles:
    fl_angle = latest_modules["front_left"]
    fr_angle = latest_modules["front_right"]
    bl_angle = latest_modules["back_left"]
    br_angle = latest_modules["back_right"]

    # We can treat the robot center as (0, 0) in the local frame.
    # The square corners in local frame (W/2, H/2, etc.)
    half = ROBOT_SIZE / 2.0
    # We'll define a line strip that goes around the perimeter:
    local_corners = [
        ( half,  half),
        ( half, -half),
        (-half, -half),
        (-half,  half),
        ( half,  half),
    ]

    # Instead of se2_transform, use to_global for each local corner:
    global_corners = []
    for (cx, cy) in local_corners:
        gx, gy = to_global(cx, cy, x, y, theta)
        global_corners.append((gx, gy))

    # Log a single line strip for the robot’s perimeter:
    rr.log(
        "robot_outline",
        rr.LineStrips2D(
            [global_corners],  # one polyline
            colors=[[128, 128, 128]],
            radii=0.002,   # thick enough to be visible
        ),
    )

    # Also log an arrow to indicate the robot's heading in the global frame
    # We'll make the arrow originate at the center (x, y) and have length = half:
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

    # Now, each module is at a corner location in the local frame (or offset).
    # Let's assume swerve modules are at the same corners as the square above:
    # front-left  => (half, half)
    # front-right => (half, -half)
    # back-left   => (-half, half)
    # back-right  => (-half, -half)
    # We'll draw an arrow from each module’s position outward in the direction (theta + module_angle).

    # (In reality, you may have slightly different offsets for each module center.)
    # For demo, we’ll show the module arrow with length = 0.05 m to see orientation clearly.
    module_arrow_len = 0.05

    def log_module_arrow(offset_x, offset_y, module_angle, name):
        # transform local offset to global “origin”:
        mod_origin = to_global(offset_x, offset_y, x, y, theta)
        # direction is robot heading + module angle
        total_angle = theta + module_angle
        vx = module_arrow_len * math.cos(total_angle)
        vy = module_arrow_len * math.sin(total_angle)

        rr.log(
            name,
            rr.Arrows2D(
                origins=[mod_origin],
                vectors=[[vx, vy]],
                colors=[[0, 255, 0]],
                labels=[name],
                show_labels=[False],

                radii=0.003,
            ),
        )

    # front-left
    log_module_arrow( half,  half, fl_angle, "front_left_module")
    # front-right
    log_module_arrow( half, -half, fr_angle, "front_right_module")
    # back-left
    log_module_arrow(-half,  half, bl_angle, "back_left_module")
    # back-right
    log_module_arrow(-half, -half, br_angle, "back_right_module")

def update_rerun_viz_3d():
    """
    Log a 3D bounding box for the robot, a pinned transform for a camera,
    and 3D arrows for each swerve module’s orientation.
    """
    x = latest_odom["x"]
    y = latest_odom["y"]
    theta = latest_odom["theta"] - np.pi / 2.0  # in radians

    # Robot center in global coordinates:
    # We'll model as a 3D box with half-sizes of (ROBOT_SIZE/2, ROBOT_SIZE/2, ROBOT_HEIGHT/2).
    half_x = ROBOT_SIZE / 2.0
    half_y = ROBOT_SIZE / 2.0
    half_z = ROBOT_HEIGHT / 2.0

    # 1) Construct yaw-based quaternion around Z:
    half_angle = theta / 2.0
    s = math.sin(half_angle)
    c = math.cos(half_angle)

    # 2) Use partial alpha color for a translucent robot box, fill_mode="solid":
    #    wireframe is not yet supported, so we'll do a partially transparent solid.
    #    RGBA => [0, 255, 255, 128] means neon cyan + 50% alpha.
    rr.log(
        "robot/base",
        rr.Boxes3D(
            centers=[[x, y, half_z]],
            half_sizes=[[half_x, half_y, half_z]],
            quaternions=[[0.0, 0.0, s, c]],
            colors=[[0, 255, 255, 128]],  # RGBA with partial alpha
            fill_mode="solid",
        ),
    )

    # Build rotation matrices for yaw, pitch, roll:
    pitch_deg = 0.0
    pitch_rad = math.radians(pitch_deg)

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

    # Add a roll of +90° around X
    roll_deg = -90.0
    roll_rad = math.radians(roll_deg)
    roll_mat = np.array([
        [1.0,         0.0,          0.0],
        [0.0,  math.cos(roll_rad), -math.sin(roll_rad)],
        [0.0,  math.sin(roll_rad),  math.cos(roll_rad)],
    ], dtype=float)



    # Compose them in an order, e.g. yaw -> pitch -> roll:
    final_rot = yaw_mat @ pitch_mat @ roll_mat

    axis, rot_angle = axis_angle_from_matrix(final_rot)

    rr.log(
        "robot/camera",
        rr.Transform3D(
            translation=[x, y, ROBOT_HEIGHT],
            rotation=RotationAxisAngle(axis=axis, angle=Angle(rad=rot_angle)),
        ),
    )

    # Log the pinhole:
    fx, fy = 597.19, 598.65
    w, h   = 1920, 1080
    rr.log("robot/camera", rr.Pinhole(focal_length=(fx, fy), width=w, height=h,image_plane_distance=1.0))

    # Now define each wheel's local offset, then log a 3D arrow for its module angle:
    def log_module_arrow_3d(offset_x, offset_y, module_angle, name):
        # Module center in global, ignoring height offset for simplicity:
        # The top level rotation is already applied to the robot base, so we
        # can re-apply if you want to physically place them in world coords:
        # A quick approach is to treat the modules as a separate object each with
        # its own transform, but below we just do an Arrows3D in the global
        # frame for clarity.

        # transform local offset to global:
        cosT = math.cos(theta)
        sinT = math.sin(theta)
        gx = x + (offset_x * cosT - offset_y * sinT)
        gy = y + (offset_x * sinT + offset_y * cosT)
        gz = 0.0  # assume on ground

        # direction is robot heading + module angle:
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

    fl_angle = latest_modules["front_left"]
    fr_angle = latest_modules["front_right"]
    bl_angle = latest_modules["back_left"]
    br_angle = latest_modules["back_right"]

    # front-left
    log_module_arrow_3d( half_x,  half_y, fl_angle, "front_left_module")
    # front-right
    log_module_arrow_3d( half_x, -half_y, fr_angle, "front_right_module")
    # back-left
    log_module_arrow_3d(-half_x,  half_y, bl_angle, "back_left_module")
    # back-right
    log_module_arrow_3d(-half_x, -half_y, br_angle, "back_right_module")

def image_listener(sample):
    """
    Decode image from raw bytes (ZBytes => bytes) and log to Rerun.
    """
    np_data = np.frombuffer(sample.payload.to_bytes(), dtype=np.uint8)
    received_img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    if received_img is not None:
        # 5) Log directly on "robot/camera" so the pinhole & the image are in the same entity:
        rr.log("robot/camera", rr.Image(received_img))

def poses_listener(sample):
    """
    Convert JSON string into Python objects for apriltag poses and print them.
    """
    poses_str = sample.payload.to_string()
    poses_data = json.loads(poses_str)
    print(f"Received {len(poses_data)} tag(s):")
    for pose_info in poses_data:
        print(f"  ID {pose_info['tag_id']} => SE3: {pose_info['SE3']}")


        # 1) Build a 4x4 SE3 transform for the camera in global coords.
        #    Assume camera yaw = robot's yaw, and z=0.254 m (~10 inches).
        robot_theta = latest_odom["theta"]
        cx = latest_odom["x"]
        cy = latest_odom["y"]
        cz = 0.254 # 10 inches

        CamGlobal = np.eye(4)
        CamGlobal[0, 3] = cx
        CamGlobal[1, 3] = cy
        CamGlobal[2, 3] = cz
        CamGlobal[0, 0] =  np.cos(robot_theta)
        CamGlobal[0, 1] = -np.sin(robot_theta)
        CamGlobal[1, 0] =  np.sin(robot_theta)
        CamGlobal[1, 1] =  np.cos(robot_theta)

        # 2) Convert the tag's SE3 pose from JSON and reshape into a 4x4 np.array
        tag_in_cam = np.array(pose_info["SE3"], dtype=float).reshape((4, 4))

        # 3) Compute the tag’s pose in the global frame: tag_global = CamGlobal @ tag_in_cam
        tag_global = CamGlobal @ tag_in_cam

        # 4) Extract translation & rotation
        tx, ty, tz = tag_global[0, 3], tag_global[1, 3], tag_global[2, 3]
        rotation_mat = tag_global[:3, :3]
        axis, angle = axis_angle_from_matrix(rotation_mat)

        # 5) Log a 3D box for the tag in Rerun
        #    You can log a purely static box or chain a Transform3D:
        half_sz = TAG_SIZE / 2.0  # half side length of the tag
        rr.log(
            f"robot/tag_{pose_info['tag_id']}",
            rr.Transform3D(
                translation=[tx, ty, tz],
                rotation=RotationAxisAngle(axis=axis, angle=Angle(rad=angle)),
            ),
        )
        rr.log(
            f"robot/tag_{pose_info['tag_id']}/box",
            rr.Boxes3D(
                centers=[[0, 0, 0]],
                half_sizes=[[half_sz, half_sz, 0.01]],  # small thickness
                colors=[[255, 255, 0]],
                fill_mode="solid",
            ),
        )
        # --- NEW CODE END ---


def main():
    # Initialize rerun
    rr.init("my_swerve_rerun", spawn=True)

    # Initialize pygame for joystick
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No joystick detected!")
        return
    joy = pygame.joystick.Joystick(0)
    joy.init() 

    # Open Zenoh session
    session = zenoh.open(zenoh.Config())
    vel_pub = session.declare_publisher(VELOCITY_KEY)
    zero_pub = session.declare_publisher(ZERO_HEADING_KEY)

    # Subscribe to robot data
    _ = session.declare_subscriber(MEASURED_TWIST_KEY, measured_twist_listener)
    _ = session.declare_subscriber(ODOMETRY_KEY, odom_listener)
    _ = session.declare_subscriber(WHEEL_VELOCITIES_KEY, wheel_velocities_callback)
    _ = session.declare_subscriber(MODULE_ANGLES_KEY, module_angles_callback)
    _ = session.declare_subscriber(LIDAR_SCAN_KEY, lidar_callback)

    # --- NEW SUBSCRIBERS INTEGRATED FROM APRILTAG_SUBSCRIBER ---
    session.declare_subscriber("robot/camera/undistorted", image_listener)
    session.declare_subscriber("robot/camera/tag_poses", poses_listener)
    # -----------------------------------------------------------

    # Send zero velocity at start
    vel_pub.put(json.dumps({"vx": 0.0, "vy": 0.0, "omega": 0.0}))

    # Create a context object for the external autonomous plugin to access
    robot_ctx = RobotContext(
        vel_pub=vel_pub,
        latest_odom=latest_odom, 
        max_speed=max_speed, 
        max_omega_deg=max_omega_deg
    )

    clock = pygame.time.Clock()
    running = True
    while True:
        # DEBUG: print how many axes we have:
        num_axes = joy.get_numaxes()

        # Print each axis raw value:
        axes = [joy.get_axis(i) for i in range(num_axes)]

        pygame.event.pump()

        # If the CROSS button is pressed and we aren't already in autonomous, launch it
        cross_pressed = joy.get_button(CROSS_BUTTON_INDEX)
        if cross_pressed and not autonomous_running:
            threading.Thread(target=autonomous_motion, args=(robot_ctx,), daemon=True).start()

        circle_pressed = joy.get_button(CIRCLE_BUTTON_INDEX)
        if circle_pressed:
            zero_pub.put("zero")

        # Read joystick, apply deadband
        lx = apply_deadband(normalize_axis(joy.get_axis(LEFT_X_AXIS)))
        ly = apply_deadband(normalize_axis(joy.get_axis(LEFT_Y_AXIS)))
        rx = apply_deadband(normalize_axis(joy.get_axis(RIGHT_X_AXIS)))

        # If autonomous is running but joystick is at (0,0,0), skip sending gamepad commands
        if not autonomous_running or (lx != 0.0 or ly != 0.0 or rx != 0.0):
            # Convert axes to velocity commands
            vx = -ly * max_speed
            vy = -lx * max_speed
            omega = -rx * max_omega_deg

            cmd = json.dumps({"vx": vx, "vy": vy, "omega": omega})
            vel_pub.put(cmd)

        # Re-log the 3D visualization each loop:
        update_rerun_viz_3d()
        clock.tick(50)  # 50 Hz

    session.close()
    pygame.quit()

if __name__ == "__main__":
    main()
