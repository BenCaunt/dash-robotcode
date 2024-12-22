import json
import time
import zenoh
import pygame
import math
import threading

# ---- NEW IMPORTS FOR RERUN:
import rerun as rr
import numpy as np

VELOCITY_KEY = "robot/control/velocity"
ZERO_HEADING_KEY = "robot/control/zero_heading"
MEASURED_TWIST_KEY = "robot/observed/twist"
ODOMETRY_KEY = "robot/odom"
WHEEL_VELOCITIES_KEY = "robot/observed/wheel_velocities"
MODULE_ANGLES_KEY = "robot/observed/module_angles"

LEFT_X_AXIS = 0
LEFT_Y_AXIS = 1
RIGHT_X_AXIS = 2
CIRCLE_BUTTON_INDEX = 1
CROSS_BUTTON_INDEX = 0  # For example, set to whichever index CROSS should be

# Robot dimensions in meters (10 inch = 0.254 m).
# We'll model the robot as a square for simplicity:
ROBOT_SIZE = 0.254

# Track global states for visualization:
latest_odom = {"x": 0.0, "y": 0.0, "theta": 0.0}
latest_modules = {
    "front_left": 0.0,
    "front_right": 0.0,
    "back_left": 0.0,
    "back_right": 0.0,
}

autonomous_running = False

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

    # SE(2) transform each local corner to global coordinates:
    def se2_transform(px, py, rx, ry, rtheta):
        cos_t = math.cos(rtheta)
        sin_t = math.sin(rtheta)
        gx = rx + cos_t*px - sin_t*py
        gy = ry + sin_t*px + cos_t*py
        return (gx, gy)

    global_corners = [se2_transform(cx, cy, x, y, theta) for (cx, cy) in local_corners]

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
        mod_origin = se2_transform(offset_x, offset_y, x, y, theta)
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

def autonomous_motion(vel_pub):
    global autonomous_running
    autonomous_running = True
    print("Starting autonomous motion...")
    for _ in range(40):
        vel_pub.put(json.dumps({"vx": 0.40, "vy": 0.0, "omega": 0.0}))
        time.sleep(0.05)
    vel_pub.put(json.dumps({"vx": 0.0, "vy": 0.0, "omega": 0.0}))
    print("Autonomous motion complete!")
    autonomous_running = False

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

    # Control scaling
    max_speed = 0.5        # m/s
    max_omega_deg = 240.0  # deg/s

    # Send zero velocity at start
    vel_pub.put(json.dumps({"vx": 0.0, "vy": 0.0, "omega": 0.0}))

    clock = pygame.time.Clock()
    running = True
    while running:
        pygame.event.pump()

        # If the CROSS button is pressed and we aren't already in autonomous, launch it
        cross_pressed = joy.get_button(CROSS_BUTTON_INDEX)
        if cross_pressed and not autonomous_running:
            threading.Thread(target=autonomous_motion, args=(vel_pub,), daemon=True).start()

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
            print(cmd)
            vel_pub.put(cmd)

        clock.tick(50)  # 50 Hz

    session.close()
    pygame.quit()

if __name__ == "__main__":
    main()
