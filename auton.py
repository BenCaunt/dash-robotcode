import time
import json
import numpy as np
import threading
from utils import angle_wrap

autonomous_running = False

class RobotContext:
    """
    Holds references/publishers for autonomous routines to access.
    """
    def __init__(self, vel_pub, latest_odom, max_speed, max_omega_deg):
        self.vel_pub = vel_pub
        self.latest_odom = latest_odom
        self.max_speed = max_speed
        self.max_omega_deg = max_omega_deg

def autonomous_motion(context: RobotContext):
    """
    Example routine that drives the robot toward (0,0,0).
    Illustrates an external 'plugin' routine that can access the context.
    """
    global autonomous_running
    autonomous_running = True
    print("Starting autonomous motion...")

    # Similar logic as before, but referencing context attributes
    trans_gain = 2.5
    omega_gain = 200.0
    is_done = False
    time_since_done = 0.0

    while not is_done and time_since_done < 0.5:
        dx = 0.0 - context.latest_odom["x"]
        dy = 0.0 - context.latest_odom["y"]
        theta = angle_wrap(0.0 - context.latest_odom["theta"])

        vx = np.clip(dx * trans_gain, -context.max_speed, context.max_speed)
        vy = np.clip(dy * trans_gain, -context.max_speed, context.max_speed)
        omega = np.clip(theta * omega_gain, -context.max_omega_deg, context.max_omega_deg)
        context.vel_pub.put(json.dumps({"vx": vx, "vy": vy, "omega": omega}))
        time.sleep(0.05)

        # Update 'is_done' logic
        at_target = (np.linalg.norm([dx, dy]) < 0.1 and abs(theta) < np.deg2rad(5.0))
        if at_target:
            time_since_done += 0.05
        else:
            time_since_done = 0.0
        is_done = (time_since_done >= 0.5)

    # Stop the robot
    context.vel_pub.put(json.dumps({"vx": 0.0, "vy": 0.0, "omega": 0.0}))
    autonomous_running = False
    print("Autonomous motion complete.")
