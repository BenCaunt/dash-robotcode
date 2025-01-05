from pupil_apriltags import Detector
import cv2
import json
import numpy as np
import zenoh
from zenoh import Config
import time

from constants import (
    TAG_SIZE,
    CAMERA_UNDISTORTED_KEY,
    CAMERA_TAG_POSES_KEY
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
    
    # Compute undistortion and rectification maps (similar to undistort_example.py)
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (image_width, image_height), 1, (image_width, image_height)
    )
    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (image_width, image_height), cv2.CV_32FC1
    )

    # Extract camera parameters from the camera matrix
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
        
    

    # Set up AprilTag detector (without pose parameters)
    detector = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=True,
        decode_sharpening=0.25,
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

            # Undistort and prepare frame for detection
            preprocess_start = time.perf_counter_ns()
            undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
            preprocess_time = (time.perf_counter_ns() - preprocess_start) / 1e6

            # Detect AprilTags
            detection_start = time.perf_counter_ns()
            detections = []
            detection_time = (time.perf_counter_ns() - detection_start) / 1e6

            # Draw detections
            drawing_start = time.perf_counter_ns()
            for detection in detections:
                corners = detection.corners
                for i in range(4):
                    pt1 = (int(corners[i][0]), int(corners[i][1]))
                    pt2 = (int(corners[(i + 1) % 4][0]), int(corners[(i + 1) % 4][1]))
                    cv2.line(undistorted, pt1, pt2, (0, 255, 0), 2)

                cX, cY = int(detection.center[0]), int(detection.center[1])
                cv2.circle(undistorted, (cX, cY), 5, (0, 0, 255), -1)
                cv2.putText(undistorted, f"ID: {detection.tag_id}", (cX - 10, cY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            drawing_time = (time.perf_counter_ns() - drawing_start) / 1e6

            # Publish to Zenoh
            publish_start = time.perf_counter_ns()
            success, buffer = cv2.imencode('.jpg', undistorted)
            if success:
                z_session.put(CAMERA_UNDISTORTED_KEY, buffer.tobytes())

            tag_poses = []
            for detection in detections:
                SE3 = np.eye(4)
                SE3[:3, :3] = detection.pose_R
                SE3[:3, 3] = detection.pose_t.flatten()
                
                transformation = np.array([
                    [0,0,1,0],
                    [-1,0,0,0],
                    [0,-1,0,0],
                    [0,0,0,1]
                ])
                tag_SE3 = transformation @ SE3
                
                tag_poses.append({
                    "tag_id": detection.tag_id,
                    "SE3": tag_SE3.tolist()
                })
            z_session.put(CAMERA_TAG_POSES_KEY, json.dumps(tag_poses))
            publish_time = (time.perf_counter_ns() - publish_start) / 1e6

            total_time = (time.perf_counter_ns() - loop_start) / 1e6
            print(f"\nTiming (ms):")
            print(f"Frame Capture: {capture_time:.1f}")
            print(f"Preprocessing: {preprocess_time:.1f}")
            print(f"Tag Detection: {detection_time:.1f}")
            print(f"Drawing     : {drawing_time:.1f}")
            print(f"Publishing  : {publish_time:.1f}")
            print(f"Total Loop  : {total_time:.1f}")

            if not HEADLESS:
                cv2.imshow("Undistorted + AprilTag Detection", undistorted)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
