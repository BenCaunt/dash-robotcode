import time
import threading
from constants import DASH_MOVEMENT_CONSTRAINT
from auton import autonomous_motion, RobotContext
from visualization import RobotVisualizer
from robot_client import RobotClient
from gamepad import GamepadController
import pygame

def main():
    # Initialize components
    visualizer = RobotVisualizer()
    robot = RobotClient(visualizer)
    try:
        gamepad = GamepadController()
    except RuntimeError as e:
        print(f"Failed to initialize gamepad: {e}")
        robot.close()
        return

    # Retrieve control scaling from constraints
    max_speed = DASH_MOVEMENT_CONSTRAINT.max_velocity.vx
    max_omega_deg = DASH_MOVEMENT_CONSTRAINT.max_velocity.w

    # Send zero velocity at start
    robot.send_velocity(0.0, 0.0, 0.0)

    clock = pygame.time.Clock()
    try:
        while True:
            # DEBUG: print raw axes
            axes = gamepad.get_raw_axes()

            # If the CROSS button is pressed and we aren't already in autonomous, launch it
            if gamepad.is_cross_pressed() and not robot.autonomous_running:
                threading.Thread(target=autonomous_motion, args=(robot,), daemon=True).start()

            if gamepad.is_circle_pressed():
                robot.zero_heading()

            # Get movement command from gamepad
            vx, vy, omega = gamepad.get_movement_command(max_speed, max_omega_deg)

            # If autonomous is running but joystick is at (0,0,0), skip sending gamepad commands
            if not robot.autonomous_running or (vx != 0.0 or vy != 0.0 or omega != 0.0):
                robot.send_velocity(vx, vy, omega)

            clock.tick(50)  # 50 Hz

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        robot.close()

if __name__ == "__main__":
    main()
