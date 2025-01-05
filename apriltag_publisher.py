import cv2
import json
import numpy as np
import zenoh
from zenoh import Config
import time

from constants import (
    CAMERA_UNDISTORTED_KEY,
)

HEADLESS = True

def main():
    # Load calibration data
    with open("camera_calibration/cam_calibration.json", "r") as f:
        calibration_data = json.load(f)

    camera_matrix = np.array(calibration_data["camera_matrix"])
    dist_coeffs = np.array(calibration_data["dist_coeffs"])
    image_width = calibration_data["image_width"]
    image_height = calibration_data["image_height"]

    # Initialize camera capture
    cap = cv2.VideoCapture(0)

    # Attempt to set 1920x1080 resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Compute undistortion and rectification maps
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (image_width, image_height), 1, (image_width, image_height)
    )
    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (image_width, image_height), cv2.CV_32FC1
    )

    # Initialize Zenoh session with config
    with zenoh.open(Config()) as z_session:
        print("Press 'q' to quit.")
        while True:
            loop_start = time.perf_counter_ns()
            
            # Capture frame
            capture_start = time.perf_counter_ns()
            ret, frame = cap.read()
            capture_time = (time.perf_counter_ns() - capture_start) / 1e6  # Convert to ms
            
            if not ret:
                print("Failed to grab frame")
                break

            # Undistort frame
            preprocess_start = time.perf_counter_ns()
            undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            preprocess_time = (time.perf_counter_ns() - preprocess_start) / 1e6

            # Publish undistorted image
            publish_start = time.perf_counter_ns()
            success, buffer = cv2.imencode('.jpg', undistorted)
            if success:
                z_session.put(CAMERA_UNDISTORTED_KEY, buffer.tobytes())
            publish_time = (time.perf_counter_ns() - publish_start) / 1e6

            total_time = (time.perf_counter_ns() - loop_start) / 1e6
            print(f"\nTiming (ms):")
            print(f"Frame Capture: {capture_time:.1f}")
            print(f"Preprocessing: {preprocess_time:.1f}")
            print(f"Publishing  : {publish_time:.1f}")
            print(f"Total Loop  : {total_time:.1f}")

            if not HEADLESS:
                cv2.imshow("Undistorted", undistorted)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
