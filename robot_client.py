import json
import cv2
import numpy as np
import zenoh
from pupil_apriltags import Detector
from constants import (
    VELOCITY_KEY, ZERO_HEADING_KEY, MEASURED_TWIST_KEY,
    ODOMETRY_KEY, WHEEL_VELOCITIES_KEY, MODULE_ANGLES_KEY,
    LIDAR_SCAN_KEY, CAMERA_UNDISTORTED_KEY, CAMERA_TAG_POSES_KEY,
    TAG_SIZE
)

class RobotClient:
    def __init__(self, visualizer=None):
        # Initialize Zenoh session
        self.session = zenoh.open(zenoh.Config())
        
        # State storage
        self.latest_odom = {"x": 0.0, "y": 0.0, "theta": 0.0}
        self.latest_modules = {
            "front_left": 0.0,
            "front_right": 0.0,
            "back_left": 0.0,
            "back_right": 0.0,
        }
        
        # Store visualizer reference
        self.visualizer = visualizer
        
        # Load camera calibration for AprilTag detection
        with open("camera_calibration/cam_calibration.json", "r") as f:
            calibration_data = json.load(f)
        camera_matrix = np.array(calibration_data["camera_matrix"])
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]

        # Initialize AprilTag detector
        self.detector = Detector(
            families="tag36h11",
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=True,
            decode_sharpening=0.25,
        )
        
        # Publishers
        self.vel_pub = self.session.declare_publisher(VELOCITY_KEY)
        self.zero_pub = self.session.declare_publisher(ZERO_HEADING_KEY)
        self.tag_poses_pub = self.session.declare_publisher(CAMERA_TAG_POSES_KEY)
        
        # Subscribers
        self._setup_subscribers()

    def _setup_subscribers(self):
        """Setup all the zenoh subscribers"""
        self.session.declare_subscriber(MEASURED_TWIST_KEY, self._measured_twist_callback)
        self.session.declare_subscriber(ODOMETRY_KEY, self._odom_callback)
        self.session.declare_subscriber(WHEEL_VELOCITIES_KEY, self._wheel_velocities_callback)
        self.session.declare_subscriber(MODULE_ANGLES_KEY, self._module_angles_callback)
        self.session.declare_subscriber(LIDAR_SCAN_KEY, self._lidar_callback)
        self.session.declare_subscriber(CAMERA_UNDISTORTED_KEY, self._image_callback)
        self.session.declare_subscriber(CAMERA_TAG_POSES_KEY, self._tag_poses_callback)

    def send_velocity(self, vx, vy, omega):
        """Send velocity command to robot"""
        self.vel_pub.put(json.dumps({"vx": vx, "vy": vy, "omega": omega}))

    def zero_heading(self):
        """Send zero heading command"""
        self.zero_pub.put("zero")

    def close(self):
        """Close the zenoh session"""
        self.session.close()

    def _measured_twist_callback(self, sample):
        """Handle measured twist data"""
        try:
            data = json.loads(sample.payload.to_string())
            # Store or process twist data if needed
        except Exception as e:
            print(f"Failed to parse measured twist: {e}")

    def _odom_callback(self, sample):
        """Handle odometry data"""
        try:
            data = json.loads(sample.payload.to_string())
            self.latest_odom["x"] = data["x"]
            self.latest_odom["y"] = data["y"]
            self.latest_odom["theta"] = data["theta"]
            
            if self.visualizer:
                self.visualizer.update_3d(self.latest_odom, self.latest_modules)
        except Exception as e:
            print(f"Failed to parse odom: {e}")

    def _wheel_velocities_callback(self, sample):
        """Handle wheel velocities data"""
        try:
            data = json.loads(sample.payload.to_string())
            if self.visualizer:
                self.visualizer.log_wheel_velocities(data)
        except Exception as e:
            print(f"Failed to parse wheel velocities: {e}")

    def _module_angles_callback(self, sample):
        """Handle module angles data"""
        try:
            data = json.loads(sample.payload.to_string())
            self.latest_modules["front_left"] = data["front_left"]
            self.latest_modules["front_right"] = data["front_right"]
            self.latest_modules["back_left"] = data["back_left"]
            self.latest_modules["back_right"] = data["back_right"]
            
            if self.visualizer:
                self.visualizer.log_module_angles(data)
                self.visualizer.update_3d(self.latest_odom, self.latest_modules)
        except Exception as e:
            print(f"Failed to parse module angles: {e}")

    def _lidar_callback(self, sample):
        """Handle lidar scan data"""
        try:
            data = json.loads(sample.payload.to_string())
            if self.visualizer:
                self.visualizer.log_lidar_scan(data, self.latest_odom)
        except Exception as e:
            print(f"Failed to parse lidar scan: {e}")

    def _image_callback(self, sample):
        """Handle camera image data and perform AprilTag detection"""
        try:
            # Decode image
            np_data = np.frombuffer(sample.payload.to_bytes(), dtype=np.uint8)
            received_img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
            if received_img is None:
                return

            # Start timing
            detection_start = time.perf_counter_ns()

            # Convert to grayscale for AprilTag detection
            gray = cv2.cvtColor(received_img, cv2.COLOR_BGR2GRAY)

            # Detect AprilTags
            detections = self.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(self.fx, self.fy, self.cx, self.cy),
                tag_size=TAG_SIZE
            )

            detection_time = (time.perf_counter_ns() - detection_start) / 1e6
            
            # Draw detections if visualizer is present
            if self.visualizer:
                img_with_detections = received_img.copy()
                for detection in detections:
                    corners = detection.corners
                    for i in range(4):
                        pt1 = (int(corners[i][0]), int(corners[i][1]))
                        pt2 = (int(corners[(i + 1) % 4][0]), int(corners[(i + 1) % 4][1]))
                        cv2.line(img_with_detections, pt1, pt2, (0, 255, 0), 2)

                    cX, cY = int(detection.center[0]), int(detection.center[1])
                    cv2.circle(img_with_detections, (cX, cY), 5, (0, 0, 255), -1)
                    cv2.putText(img_with_detections, f"ID: {detection.tag_id}", (cX - 10, cY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                self.visualizer.log_image(img_with_detections)

            # Process and publish tag poses
            tag_poses = []
            for detection in detections:
                SE3 = np.eye(4)
                SE3[:3, :3] = detection.pose_R
                SE3[:3, 3] = detection.pose_t.flatten()
                
                # map (x,y,z) -> (z,-x,y)
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

            # Publish tag poses
            self.tag_poses_pub.put(json.dumps(tag_poses))

            # Print timing info
            print(f"AprilTag Detection Time: {detection_time:.1f} ms")

        except Exception as e:
            print(f"Failed to process image: {e}")

    def _tag_poses_callback(self, sample):
        """Handle AprilTag poses data"""
        try:
            poses_data = json.loads(sample.payload.to_string())
            if self.visualizer:
                for pose_info in poses_data:
                    tag_in_cam = np.array(pose_info["SE3"], dtype=float).reshape((4, 4))
                    self.visualizer.log_apriltag(pose_info["tag_id"], tag_in_cam, self.latest_odom)
        except Exception as e:
            print(f"Failed to process tag poses: {e}") 