#!/usr/bin/python3 -B

import asyncio
import math

import numpy as np
import moteus
import moteus_pi3hat
import time
from tqdm import tqdm
from multiprocessing import JoinableQueue

AZIMUTH_RATIO = 12.0 / 75.0
DRIVE_REDUCTION = (25.0/21.0) * (15.0/45.0)

DRIVE_DIAMETER = 0.075  # 75 mm
DRIVE_CIRCUMFERENCE = DRIVE_DIAMETER * math.pi

WATCHDOG_TIMEOUT = 0.5

# GLOBAL STATE
last_recv = time.monotonic()
reference_vx = 0.0  # m/s
reference_vy = 0.0  # m/s
reference_w = 0.0  # rad/s
offset = 0.0 # rad
is_initial_angle = True
reference_heading = 0.0
gain = 0.1
angular_velocity_constant = 0.0
yaw_bias_integral = 0.0

from kinematics import robot_relative_velocity_to_twist, twist_to_wheel_speeds, WheelSpeeds, ModuleAngles
from geometry2d import Twist2dVelocity


def angle_wrap(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def calculate_swerve_angle(position: float) -> float:
    return angle_wrap(position * 2 * math.pi * AZIMUTH_RATIO)


def wheel_speed_to_motor_speed(wheel_speed: float) -> float:
    return wheel_speed / (DRIVE_CIRCUMFERENCE * DRIVE_REDUCTION)


def motor_speed_to_wheel_speed(motor_speed: float) -> float:
    return motor_speed * (DRIVE_CIRCUMFERENCE * DRIVE_REDUCTION)


def calculate_target_position_delta(reference_azimuth_angle, estimated_angle):
    angle_difference = angle_wrap(reference_azimuth_angle - estimated_angle)
    return angle_difference / (2 * math.pi * AZIMUTH_RATIO)


async def main():
    global reference_vx, reference_vy, reference_w, offset, is_initial_angle, reference_heading

    transport = moteus_pi3hat.Pi3HatRouter(
        servo_bus_map={1: [1, 2, 4], 2: [5, 6], 3: [3, 7, 8]},
    )

    azimuth_ids = [2, 4, 6, 8]
    drive_ids = [1, 3, 5, 7]

    servos = {servo_id: moteus.Controller(id=servo_id, transport=transport) for servo_id in azimuth_ids + drive_ids}

    drive_directions = {1: 1, 3: 1, 5: -1, 7: -1}

    # Stop all servos
    results = await transport.cycle([x.make_stop(query=True) for x in servos.values()])
    # print(results)
    initial_module_positions = {
        result.id: result.values[moteus.Register.POSITION] for result in results if result.id in azimuth_ids
    }

    offset = 0.0

    yaw = 0.0

    measured_module_positions = {2: 0.0, 4: 0.0, 6: 0.0, 8: 0.0}
    module_scaling = {2: 1.0, 4: 1.0, 6: 1.0, 8: 1.0}
    module_inversions = {2: True, 4: False, 6: False, 8: False}

    try:
        loop_start = time.monotonic()
        dt = 0.005

        while True:
            dt = time.monotonic() - loop_start
            loop_start = time.monotonic()

            if loop_start - last_recv > WATCHDOG_TIMEOUT:
                reference_vx = 0.001
                reference_vy = 0.0
                reference_w = 0.0

            # if not command_queue.empty():
            #     command = command_queue.get()
            #     global reference_vx, reference_vy, reference_w

            #     reference_vx = command.vx
            #     reference_vy = command.vy
            #     reference_w = command.omega

            #     command_queue.task_done()

            #     print(f"reference_vx: {reference_vx}, reference_vy: {reference_vy}, reference_w: {reference_w}")
            reference_heading = reference_heading + reference_w * dt
            heading_error = angle_wrap(reference_heading - -yaw)
            heading_gain = 0.0
            if np.abs(reference_w) > 0.1:
                reference_heading = -yaw
                heading_gain = 0.0
            reference = Twist2dVelocity(reference_vx, reference_vy, reference_w + heading_error * heading_gain)
            wheel_speeds, module_angles = robot_relative_velocity_to_twist(reference, dt, -(yaw + np.pi/2)) # here for 'yaw'
            # print(f"Yaw angle (rad): {yaw} | Yaw angle (deg): {math.degrees(yaw)}")
            # print(module_angles.to_list_degrees())
            print(f"heading error {heading_error}")
            commands = []
            for id in azimuth_ids:
                current_angle = calculate_swerve_angle(measured_module_positions[id]) - calculate_swerve_angle(
                    initial_module_positions[id]
                )
                current_angle = angle_wrap(current_angle)
                target_angle = module_angles.from_id(id)
                # this makes it so + is ccw with module rotation.
                target_angle = -angle_wrap(target_angle)

                # if module_inversions[id]:
                #     target_angle = angle_wrap(np.pi - target_angle)

                error = angle_wrap(target_angle - current_angle)
                # print(f"error: {math.degrees(error)}")

                module_scaling[id] = np.cos(np.clip(error, -np.pi / 2, np.pi / 2))

                # if id == 2 and module_inversions[id]:
                #     print(f"inverting module {id}")

                if abs(error) > np.pi / 2:
                    module_inversions[id] = not module_inversions[id]

                target_position_delta = calculate_target_position_delta(target_angle, current_angle)

                commands.append(
                    servos[id].make_position(
                        position=measured_module_positions[id] + target_position_delta,
                        velocity=0.0,
                        maximum_torque=1.7,
                        velocity_limit=230.0,
                        accel_limit=120.0,
                        query=True,
                    )
                )

            for id in drive_ids:
                sign = 1.0

                # if module_inversions[id + 1]:
                #     sign = -1.0

                commands.append(
                    servos[id].make_position(
                        position=math.nan,
                        velocity=module_scaling[id + 1]
                        * sign
                        * wheel_speed_to_motor_speed(wheel_speeds.from_id(id))
                        * drive_directions[id],
                        maximum_torque=1.0 * 0.25,
                        query=True,
                    )
                )

            results = await transport.cycle(commands, request_attitude=True)

            imu_result = [x for x in results if x.id == -1 and isinstance(x, moteus_pi3hat.CanAttitudeWrapper)][0]

            if is_initial_angle:
                offset = imu_result.euler_rad.yaw
                yaw_bias_integral = 0.0
                angular_velocity_constant = np.deg2rad(imu_result.rate_dps.z)

                is_initial_angle = False

            yaw_bias_integral += angular_velocity_constant * dt

            yaw = angle_wrap(imu_result.euler_rad.yaw - (offset - yaw_bias_integral))

            measured_module_positions = {
                result.id: result.values[moteus.Register.POSITION] for result in results if result.id in azimuth_ids
            }

            await asyncio.sleep(0.005)

    except KeyboardInterrupt:
        print("\nStopping all servos...")
        await transport.cycle([x.make_stop() for x in servos.values()])


if __name__ == "__main__":
    asyncio.run(main())
