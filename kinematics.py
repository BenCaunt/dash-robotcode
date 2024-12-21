import math
from typing import List, Tuple
from geometry2d import Transform2d, Twist2d, Twist2dVelocity, Vector2d
import numpy as np
from dataclasses import dataclass

wheel_base = 139.00000 / 1000.0  # 139 mm
track_width = 139.00000 / 1000.0  # 139 mm


fl_pos = Vector2d(wheel_base / 2, track_width / 2)  # v1
fr_pos = Vector2d(wheel_base / 2, -track_width / 2)  # v2
bl_pos = Vector2d(-wheel_base / 2, track_width / 2)  # v3
br_pos = Vector2d(-wheel_base / 2, -track_width / 2)  # v4

prev_twist = Twist2dVelocity(0, 0, 0)

@dataclass
class WheelSpeeds:
    front_left: float
    front_right: float
    back_left: float
    back_right: float

    def from_id(self, id: int) -> float:
        assert id % 2 == 1
        if id == 1:
            return self.front_left
        elif id == 3:
            return self.front_right
        elif id == 5:
            return self.back_left
        elif id == 7:
            return self.back_right
        else:
            raise ValueError(f"Invalid id: {id}")


@dataclass
class ModuleAngles:
    front_left_angle: float
    front_right_angle: float
    back_left_angle: float
    back_right_angle: float

    def from_id(self, id: int) -> float:
        assert id % 2 == 0
        if id == 4:
            return self.front_left_angle
        elif id == 8:
            return self.front_right_angle
        elif id == 2:
            return self.back_left_angle
        elif id == 6:
            return self.back_right_angle
        else:
            raise ValueError(f"Invalid id: {id}")

    def to_list(self) -> List[float]:
        return [self.front_left_angle, self.front_right_angle, self.back_left_angle, self.back_right_angle]

    def to_list_degrees(self) -> List[float]:
        return [math.degrees(angle) for angle in self.to_list()]



def robot_relative_velocity_to_twist(twist: Twist2dVelocity, dt, yaw: float) -> Tuple[WheelSpeeds, ModuleAngles]:
    v = Vector2d(twist.vx, twist.vy)
    v = v.rotate(-yaw)
    twist = Twist2dVelocity(v.x, v.y, twist.w)
    return twist_to_wheel_speeds(twist, dt)

def apply_acceleration_limit(twist: Twist2dVelocity, dt: float) -> Twist2dVelocity:
    # the borrow checker should really be smart enough to figure out that this is safe in this case.  I love python.
    global prev_twist
    max_acceleration = 16.0
    max_deceleration = 16.0

    max_angular_acceleration = np.deg2rad(360 * 8)
    max_angular_deceleration = np.deg2rad(360 * 8)

    delta_twist = twist - prev_twist

    delta_twist.vx = np.clip(delta_twist.vx, -max_deceleration * dt, max_acceleration * dt)
    delta_twist.vy = np.clip(delta_twist.vy, -max_deceleration * dt, max_acceleration * dt)
    delta_twist.w = np.clip(delta_twist.w, -max_angular_deceleration * dt, max_angular_acceleration * dt)
    
    prev_twist = prev_twist + delta_twist

    return prev_twist

def twist_to_wheel_speeds(twist: Twist2dVelocity, dt: float) -> Tuple[WheelSpeeds, ModuleAngles]:

    transform = Transform2d(twist.vx * dt, twist.vy * dt, twist.w * dt)
    twist = transform.log()
    twist = Twist2dVelocity(twist.dx / dt, twist.dy / dt, twist.dyaw / dt)
    twist = apply_acceleration_limit(twist, dt) 

    twist = np.array([twist.vx, twist.vy, twist.w])
    
    # if epsilon_equals(twist[0], 0, 0.05) and epsilon_equals(twist[1], 0, 0.05) and not epsilon_equals(twist[2], 0, 0.05):
    #     theta1 = np.deg2rad(-45)
    #     theta2 = np.deg2rad(45)
    #     theta3 = np.deg2rad(-45)
    #     theta4 = np.deg2rad(45)

    #     radius = np.sqrt(wheel_base * wheel_base)  # hypotenuse divided by 2 to get distance from center to wheel
    #     # w = v / r
    #     # v = w * r
    #     v1 = twist[2] * radius
    #     v2 = twist[2] * radius
    #     v3 = twist[2] * radius
    #     v4 = twist[2] * radius
    # else:
    transition = np.array(
        [
            [1, 0, -fl_pos.y],
            [0, 1, fl_pos.x],
            [1, 0, -fr_pos.y],
            [0, 1, fr_pos.x],
            [1, 0, -bl_pos.y],
            [0, 1, bl_pos.x],
            [1, 0, -br_pos.y],
            [0, 1, br_pos.x],
        ]
    )
    speeds = np.dot(transition, twist.transpose())

    v1 = np.sqrt(speeds[0] ** 2 + speeds[1] ** 2)
    v2 = np.sqrt(speeds[2] ** 2 + speeds[3] ** 2)
    v3 = np.sqrt(speeds[4] ** 2 + speeds[5] ** 2)
    v4 = np.sqrt(speeds[6] ** 2 + speeds[7] ** 2)

    theta1 = np.arctan2(speeds[1], speeds[0])
    theta2 = np.arctan2(speeds[3], speeds[2])
    theta3 = np.arctan2(speeds[5], speeds[4])
    theta4 = np.arctan2(speeds[7], speeds[6])

    return WheelSpeeds(front_left=v1, front_right=v2, back_left=v3, back_right=v4), ModuleAngles(
        front_left_angle=theta1, front_right_angle=theta2, back_left_angle=theta3, back_right_angle=theta4
    )
    
def wheel_speeds_to_twist(speeds: WheelSpeeds, angles: ModuleAngles, dt: float) -> Twist2dVelocity:
    # Calculate the velocity components for each wheel
    v1_x = speeds.front_left * math.cos(angles.front_left_angle)
    v1_y = speeds.front_left * math.sin(angles.front_left_angle)
    v2_x = speeds.front_right * math.cos(angles.front_right_angle)
    v2_y = speeds.front_right * math.sin(angles.front_right_angle)
    v3_x = speeds.back_left * math.cos(angles.back_left_angle)
    v3_y = speeds.back_left * math.sin(angles.back_left_angle)
    v4_x = speeds.back_right * math.cos(angles.back_right_angle)
    v4_y = speeds.back_right * math.sin(angles.back_right_angle)

    # Create the inverse transition matrix
    inverse_transition = np.linalg.pinv(
        [
            [1, 0, -fl_pos.y],
            [0, 1, fl_pos.x],
            [1, 0, -fr_pos.y],
            [0, 1, fr_pos.x],
            [1, 0, -bl_pos.y],
            [0, 1, bl_pos.x],
            [1, 0, -br_pos.y],
            [0, 1, br_pos.x],
        ]
    )

    # Combine the velocities into a single vector
    wheel_velocities = np.array([v1_x, v1_y, v2_x, v2_y, v3_x, v3_y, v4_x, v4_y])

    # Calculate the twist
    twist = np.dot(inverse_transition, wheel_velocities)

    return Twist2dVelocity(twist[0] / dt, twist[1] / dt, twist[2] / dt)


if __name__ == "__main__":
    # twist = Twist2dVelocity(1.0, 0.0, 0.0)
    # speeds, angles = twist_to_wheel_speeds(twist, 0.05)

    # print(angles.to_list_degrees())
    # print(speeds.front_left)
    # print(speeds.front_right)
    # print(speeds.back_left)
    # print(speeds.back_right)
    # dt = 1.0
    # yaw = np.deg2rad(90.0)
    # tf = Transform2d(twist.vx * dt, twist.vy * dt, twist.w * dt)
    # tf = Transform2d(0, 0, yaw) * tf
    # print(tf.x, tf.y, tf.theta)
    wheel_speeds = WheelSpeeds(front_left=1.0, front_right=1.0, back_left=1.0, back_right=1.0)
    angles = ModuleAngles(front_left_angle=np.deg2rad(45), front_right_angle=np.deg2rad(45), back_left_angle=np.deg2rad(45), back_right_angle=np.deg2rad(45))
    twist = wheel_speeds_to_twist(wheel_speeds, angles, 1.0)
    print(twist.vx, twist.vy, twist.w)


def epsilon_equals(a: float, b: float, epsilon: float) -> bool:
    return abs(a - b) < epsilon

def measured_positions_to_module_angles(
    measured_positions: dict[int, float],
    initial_positions: dict[int, float] = None,
    offset_deg: float = 0.0
) -> ModuleAngles:
    """
    Convert raw measured positions (in rotations) into a ModuleAngles object (in radians),
    optionally subtracting initial 'zero' positions and applying an extra offset in degrees.
    
    Motor ID mapping from kinematics.py:
      - ID 4 -> front_left
      - ID 8 -> front_right
      - ID 2 -> back_left
      - ID 6 -> back_right
    """

    def wrap(angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def rotation_to_angle(rotation: float) -> float:
        # Same ratio as main.py
        AZIMUTH_RATIO = 12.0 / 75.0
        return wrap(rotation * 2 * math.pi * AZIMUTH_RATIO)

    def get_angle(motor_id: int) -> float:
        # Subtract initial offset if provided
        adjusted_rotation = measured_positions[motor_id]
        if initial_positions is not None:
            adjusted_rotation -= initial_positions[motor_id]
        # Convert rotation to angle
        angle = rotation_to_angle(adjusted_rotation)
        # Apply the user-specified offset in degrees
        angle -= math.radians(offset_deg)
        return wrap(angle)

    return ModuleAngles(
        front_left_angle=get_angle(4),
        front_right_angle=get_angle(8),
        back_left_angle=get_angle(2),
        back_right_angle=get_angle(6),
    )


