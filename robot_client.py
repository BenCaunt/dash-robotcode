import json
import cv2
import numpy as np
import zenoh
from constants import (
    VELOCITY_KEY, ZERO_HEADING_KEY, MEASURED_TWIST_KEY,
    ODOMETRY_KEY, WHEEL_VELOCITIES_KEY, MODULE_ANGLES_KEY,
    LIDAR_SCAN_KEY, CAMERA_UNDISTORTED_KEY, CAMERA_TAG_POSES_KEY
)
from visualization import RobotVisualizer

class RobotClient:
    def __init__(self, visualizer: RobotVisualizer):
        # Initialize Zenoh session
        self.session = zenoh.open(zenoh.Config())
        
        # State storage
        self.latest_odom = {"x": 0.0, "y": 0.0, "theta": 0.0, "timestamp": 0.0}
        self.latest_modules = {
            "front_left": 0.0,
            "front_right": 0.0,
            "back_left": 0.0,
            "back_right": 0.0,
        }
        
        # Odometry history for synchronization
        self.odom_history = []
        self.MAX_HISTORY_SIZE = 100  # Keep last 100 odometry messages
        self.MAX_TIME_DIFF = 0.1  # Maximum time difference for synchronization (100ms)
        
        # Store visualizer reference
        self.visualizer = visualizer
        
        # Publishers
        self.vel_pub = self.session.declare_publisher(VELOCITY_KEY)
        self.zero_pub = self.session.declare_publisher(ZERO_HEADING_KEY)
        
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
            self.latest_odom = data
            
            # Add to history and maintain max size
            self.odom_history.append(data)
            if len(self.odom_history) > self.MAX_HISTORY_SIZE:
                self.odom_history.pop(0)
            
            if self.visualizer:
                self.visualizer.update_3d(self.latest_odom, self.latest_modules)
        except Exception as e:
            print(f"Failed to parse odom: {e}")

    def _find_closest_odom(self, timestamp):
        """Find the odometry data closest to the given timestamp"""
        if not self.odom_history:
            return self.latest_odom
        
        # Find closest timestamp
        closest = min(self.odom_history, 
                     key=lambda x: abs(x["timestamp"] - timestamp))
        
        # Check if within acceptable time difference
        if abs(closest["timestamp"] - timestamp) > self.MAX_TIME_DIFF:
            print(f"Warning: Large time difference ({abs(closest['timestamp'] - timestamp):.3f}s) in odometry synchronization")
        
        return closest

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
        """Handle camera image data"""
        try:
            np_data = np.frombuffer(sample.payload.to_bytes(), dtype=np.uint8)
            received_img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
            if received_img is not None and self.visualizer:
                self.visualizer.log_image(received_img)
        except Exception as e:
            print(f"Failed to process image: {e}")

    def _tag_poses_callback(self, sample):
        """Handle AprilTag poses data"""
        try:
            poses_data = json.loads(sample.payload.to_string())
            if self.visualizer:
                for pose_info in poses_data:
                    timestamp = pose_info["timestamp"]
                    print(f"Received tag pose {pose_info['tag_id']} at {timestamp}")
                    tag_in_cam = np.array(pose_info["SE3"], dtype=float).reshape((4, 4))
                    
                    # Find synchronized odometry data
                    synced_odom = self._find_closest_odom(timestamp)
                    self.visualizer.log_apriltag(pose_info["tag_id"], tag_in_cam, synced_odom)
        except Exception as e:
            print(f"Failed to process tag poses: {e}") 