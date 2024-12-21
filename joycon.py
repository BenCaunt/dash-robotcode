import json
import time
import zenoh
import pygame

VELOCITY_KEY = "robot/control/velocity"
ZERO_HEADING_KEY = "robot/control/zero_heading"
MEASURED_TWIST_KEY = "robot/observed/twist"

# Adjust these indices as needed for your specific controller
LEFT_X_AXIS = 0
LEFT_Y_AXIS = 1
RIGHT_X_AXIS = 2  # may differ, check with print statements

CIRCLE_BUTTON_INDEX = 1  # Often circle is button 1 on PlayStation, verify

def apply_deadband(value, deadband=0.05):
    """
    Apply deadband to axis input. Values within +/- deadband are set to 0.
    Values outside are scaled to still use the full range.
    """
    if abs(value) < deadband:
        return 0.0
    # Scale the remaining range to still use full scale
    sign = 1.0 if value > 0 else -1.0
    scaled = (abs(value) - deadband) / (1 - deadband)
    return sign * scaled

def normalize_axis(value):
    # pygame returns axis values in [-1.0, 1.0]
    return value

def main():
    # Initialize pygame and the joystick
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

    # Subscribe to measured twist
    _ = session.declare_subscriber(MEASURED_TWIST_KEY, measured_twist_listener)

    # Control scaling:
    max_speed = 0.5      # m/s
    max_omega_deg = 240.0 # deg/s

    # Send zero velocity at start
    vel_pub.put(json.dumps({"vx": 0.0, "vy": 0.0, "omega": 0.0}))

    clock = pygame.time.Clock()

    running = True
    while running:
        # Pump and process events
        pygame.event.pump()

        # Check if circle button is pressed
        # Note: For continuous checking, you can read button states directly.
        circle_pressed = joy.get_button(CIRCLE_BUTTON_INDEX)
        if circle_pressed:
            print("Circle pressed -> zero heading")
            zero_pub.put("zero")

        # Read axes and apply deadband
        lx = apply_deadband(normalize_axis(joy.get_axis(LEFT_X_AXIS)))
        ly = apply_deadband(normalize_axis(joy.get_axis(LEFT_Y_AXIS)))
        rx = apply_deadband(normalize_axis(joy.get_axis(RIGHT_X_AXIS)))

        # Convert axes to velocity commands
        vx = -ly * max_speed    # forward on stick usually negative Y
        vy = -lx * max_speed
        omega = -rx * max_omega_deg

        # Publish updated velocity

        cmd = json.dumps({"vx": vx, "vy": vy, "omega": omega})
        print(cmd)
        vel_pub.put(cmd)

        # Limit update rate
        clock.tick(50)  # 50 Hz

    session.close()
    pygame.quit()

def measured_twist_listener(sample):
    data_str = sample.payload.to_string()
    try:
        data = json.loads(data_str)
        vx = data["vx"]
        vy = data["vy"]
        omega = data["omega"]  # rad/s
        print(f"Measured Twist: vx={vx:.3f} m/s, vy={vy:.3f} m/s, omega={omega:.3f} rad/s")
    except Exception as e:
        print(f"Failed to parse measured twist: {e}")

if __name__ == "__main__":
    main()
