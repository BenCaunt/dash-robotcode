import numpy as np
from constants import SE2Constraint, VELOCITY_KEY, ZERO_HEADING_KEY, ODOMETRY_KEY, WHEEL_VELOCITIES_KEY, MODULE_ANGLES_KEY, MEASURED_TWIST_KEY
from geometry2d import Transform2d, Twist2dVelocity
from motion import SE2Trajectory

import json
import time
import zenoh
import asyncio
import math

# We'll reuse kinematics helpers to produce module angles, etc.
from kinematics import twist_to_wheel_speeds, ModuleAngles, WheelSpeeds

class SimRobot:
    def __init__(self, initial_pose: Transform2d, robot_constraints: SE2Constraint):
        self.initial_pose = initial_pose
        self.robot_constraints = robot_constraints
        self.__CURRENT_TRUE_POSE = initial_pose # hidden
        self.__CURRENT_TRUE_VELOCITY = Twist2dVelocity(0.0, 0.0, 0.0) # hidden

# Additional states for simulation:
current_twist = Twist2dVelocity(0.0, 0.0, 0.0)   # Velocity commanded by the user (m/s, m/s, deg/s -> rad/s)
sim_pose = Transform2d(0.0, 0.0, 0.0)
zero_heading_requested = False
last_message_time = time.monotonic()

WATCHDOG_TIMEOUT = 0.5

# Zenoh session reference
session = None
odom_pub = None
wheel_vels_pub = None
module_angles_pub = None
measured_twist_pub = None

def velocity_listener(sample):
    """
    Listens for incoming velocity setpoints in JSON:
      { "vx": <float>, "vy": <float>, "omega": <float> (deg/s) }
    """
    global current_twist, last_message_time
    data_str = sample.payload.to_string()
    try:
        data = json.loads(data_str)
        # Convert omega from deg/s to rad/s
        omega_rad = math.radians(data["omega"])
        current_twist = Twist2dVelocity(data["vx"], data["vy"], omega_rad)
        last_message_time = time.monotonic()
    except Exception as e:
        print(f"Failed to parse velocity command: {e}")

def zero_heading_listener(sample):
    """
    Called when "zero_heading" is received - for the physical robot, 
    this resets IMU offset. We'll do something trivial, like re-center 
    our sim pose to zero heading.
    """
    global zero_heading_requested
    zero_heading_requested = True

async def simulation_loop(robot: SimRobot, dt=0.05):
    global sim_pose, current_twist, zero_heading_requested

    while True:
        # (Optional) Watchdog code commented out.
        # if (time.monotonic() - last_message_time) > WATCHDOG_TIMEOUT:
        #     current_twist = Twist2dVelocity(0.0, 0.0, 0.0)

        # Zero heading if commanded
        if zero_heading_requested:
            sim_pose = Transform2d(sim_pose.x, sim_pose.y, 0.0)
            zero_heading_requested = False

        # Because current_twist (vx, vy) is intended to be a GLOBAL velocity,
        # we must convert it to LOCAL velocity if we want the swerve kinematics
        # to work as if it were local. We'll rotate (vx, vy) by -sim_pose.theta:

        # In the sim, you're updating global pose directly by (vx*dt, vy*dt).
        # That’s fine, but the wheels need local velocity for angle calculation.
        pose_update = Transform2d(0.0, 0.0, -sim_pose.theta) * Transform2d(current_twist.vx * dt,
                                  current_twist.vy * dt,
                                  0.0)
        pose_update = Transform2d(pose_update.x, pose_update.y, current_twist.w * dt)
        
        print(pose_update)
        sim_pose = sim_pose * pose_update

        # Publish odometry
        odom_msg = json.dumps({
            "x": sim_pose.x,
            "y": sim_pose.y,
            "theta": sim_pose.theta
        })
        odom_pub.put(odom_msg)

        # Now compute wheel speeds and module angles using the LOCAL velocity:
        local_twist = pose_update.log()
        local_twist = Twist2dVelocity(local_twist.dx / dt, local_twist.dy / dt, local_twist.dyaw / dt)
        wheel_speeds, module_angles = twist_to_wheel_speeds(local_twist, dt)

        # Convert to JSON and publish
        wv_msg = json.dumps({
            "front_left": wheel_speeds.front_left,
            "front_right": wheel_speeds.front_right,
            "back_left": wheel_speeds.back_left,
            "back_right": wheel_speeds.back_right
        })
        wheel_vels_pub.put(wv_msg)


    
        if np.linalg.norm(np.array([current_twist.vx, current_twist.vy])) > 0.01 or np.abs(current_twist.w) > 0.01:
            ma_msg = json.dumps({
                "front_left": -module_angles.front_left_angle,
                "front_right": -module_angles.front_right_angle,
                "back_left": -module_angles.back_left_angle,
                "back_right": -module_angles.back_right_angle
            })
            module_angles_pub.put(ma_msg)

        # Also publish "measured twist" (the global velocity in your sim):
        measured_twist_msg = json.dumps({
            "vx": current_twist.vx,
            "vy": current_twist.vy,
            "omega": current_twist.w
        })
        measured_twist_pub.put(measured_twist_msg)

        await asyncio.sleep(dt)

def main():
    global session, odom_pub, wheel_vels_pub, module_angles_pub, measured_twist_pub
    # Initialize the sim robot
    robot = SimRobot(
        initial_pose=Transform2d(0.0, 0.0, 0.0),
        robot_constraints=SE2Constraint(
            max_velocity=Twist2dVelocity(1.0, 1.0, math.radians(360.0)),
            max_acceleration=Twist2dVelocity(1.0, 1.0, math.radians(360.0))
        ),
    )

    # Open Zenoh session
    session = zenoh.open(zenoh.Config())

    # Declare publishers
    odom_pub = session.declare_publisher(ODOMETRY_KEY)
    wheel_vels_pub = session.declare_publisher(WHEEL_VELOCITIES_KEY)
    module_angles_pub = session.declare_publisher(MODULE_ANGLES_KEY)
    measured_twist_pub = session.declare_publisher(MEASURED_TWIST_KEY)

    # Subscribe to velocity + zero heading
    _ = session.declare_subscriber(VELOCITY_KEY, velocity_listener)
    _ = session.declare_subscriber(ZERO_HEADING_KEY, zero_heading_listener)

    # Launch the simulation loop
    asyncio.run(simulation_loop(robot))

if __name__ == "__main__":
    main()


