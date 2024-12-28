import time
import json

import cv2
import numpy as np
import zenoh
from zenoh import Config
import rerun as rr


def image_listener(sample):
    # Decode image from raw bytes (ZBytes => bytes)
    np_data = np.frombuffer(sample.payload.to_bytes(), dtype=np.uint8)
    received_img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    if received_img is not None:
        # Log the image to Rerun
        rr.log("robot/camera/undistorted", rr.Image(received_img))


def poses_listener(sample):
    # Convert JSON string back into Python objects
    poses_str = sample.payload.to_string()
    poses_data = json.loads(poses_str)
    print(f"Received {len(poses_data)} tag(s):")
    for pose_info in poses_data:
        print(f"  ID {pose_info['tag_id']} => SE3: {pose_info['SE3']}")


def main():
    rr.init("apriltag_subscriber", spawn=True)  # Initialize Rerun logging
    # Create a default Zenoh config
    with zenoh.open(Config()) as session:
        session.declare_subscriber("robot/camera/undistorted", image_listener)
        session.declare_subscriber("robot/camera/tag_poses", poses_listener)

        print("Press Ctrl-C to exit.")
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
