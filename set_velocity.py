import time
import json
import zenoh

VELOCITY_KEY = "robot/control/velocity"

def main():
    # Open Zenoh session
    session = zenoh.open(zenoh.Config())
    pub = session.declare_publisher(VELOCITY_KEY)

    # Send forward velocity command
    forward_cmd = json.dumps({"vx": 0.25, "vy": 0.0, "omega": 0.0})
    print("Sending forward command...")
    pub.put(forward_cmd)

    # Wait for 1 second
    time.sleep(1.0)

    # Send stop command
    stop_cmd = json.dumps({"vx": 0.0, "vy": 0.0, "omega": 0.0})
    print("Sending stop command...")
    pub.put(stop_cmd)

    # Close session
    session.close()

if __name__ == "__main__":
    main()
