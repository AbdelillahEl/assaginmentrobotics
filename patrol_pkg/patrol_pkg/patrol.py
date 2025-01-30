import cv2
import mediapipe as mp
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import ReliabilityPolicy, QoSProfile

class HandGestureAndLidar(Node):
    def __init__(self):
        super().__init__('hand_gesture_lidar')

        # ROS2 publishers and subscribers
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.subscriber = self.create_subscription(
            LaserScan, 
            '/scan', 
            self.laser_callback, 
            QoSProfile(depth=10, reliability=ReliabilityPolicy.SYSTEM_DEFAULT)
        )

        self.laser_forward = 0
        self.laser_left = float('inf')
        self.laser_right = float('inf')
        self.safe_distance = 0.5
        self.cmd = Twist()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Replace camera setup with turtlebot camera stream
        self.cap = cv2.VideoCapture("http://10.2.172.103:8080/?action=stream")
        if not self.cap.isOpened():
            self.get_logger().error("Could not open turtlebot camera stream")
            raise RuntimeError("Camera stream not accessible")

        # Control variables
        self.gesture_state = "STOP"
        self.timer = self.create_timer(0.1, self.control_loop)

    def laser_callback(self, msg):
        self.laser_forward = msg.ranges[0] 
        self.laser_left = min([r for r in msg.ranges[0:45] if r != float('inf')], default=float('inf'))
        self.laser_right = min([r for r in msg.ranges[-45:] if r != float('inf')], default=float('inf'))

    def recognize_gesture(self, hand_landmarks):
        if not hand_landmarks:
            return "UNKNOWN"

        # Get landmark positions
        landmarks = hand_landmarks.landmark

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        # New gesture mappings:
        thumb_index_distance = ((thumb_tip.x - index_tip.x)**2 + 
                              (thumb_tip.y - index_tip.y)**2)**0.5

        # OK sign (thumb and index finger form a circle)
        if thumb_index_distance < 0.1:
            return "TURN_LEFT"

        # Open palm (all fingers extended)
        elif (thumb_tip.y < landmarks[3].y and 
              index_tip.y < landmarks[6].y and 
              middle_tip.y < landmarks[10].y and 
              ring_tip.y < landmarks[14].y and 
              pinky_tip.y < landmarks[18].y):
            return "STOP"

        # Thumbs up (thumb above palm, other fingers curled)
        elif (thumb_tip.y < landmarks[2].y and 
              all(landmarks[i].y > landmarks[i - 2].y for i in [8, 12, 16, 20])):
            return "FORWARD"

        # Fist (all fingers curled)
        elif all(landmarks[i].y > landmarks[i - 2].y for i in [4, 8, 12, 16, 20]):
            return "BACKWARD"

        return "UNKNOWN"

    def rescale_frame(self, frame, percent=75):
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    def control_loop(self):
        success, img = self.cap.read()
        if not success:
            return

        img = self.rescale_frame(img, 75)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                self.gesture_state = self.recognize_gesture(hand_landmarks)

        # Display gesture state
        cv2.putText(img, f"Gesture: {self.gesture_state}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Control logic with fixed speeds
        if self.gesture_state == "STOP":
            self.cmd.linear.x = 0.0
            self.cmd.angular.z = 0.0
            self.get_logger().info('STOP command sent')
        elif self.gesture_state == "FORWARD":
            if self.laser_forward < self.safe_distance:
                if self.laser_left > self.laser_right:
                    self.cmd.linear.x = 0.0
                    self.cmd.angular.z = 0.2
                    self.get_logger().info('Turning left due to obstacle')
                else:
                    self.cmd.linear.x = 0.0
                    self.cmd.angular.z = -0.2
                    self.get_logger().info('Turning right due to obstacle')
            else:
                self.cmd.linear.x = 0.2  # Fixed forward speed
                self.cmd.angular.z = 0.0
                self.get_logger().info('Moving forward')
        elif self.gesture_state == "BACKWARD":
            self.cmd.linear.x = -0.2  # Fixed backward speed
            self.cmd.angular.z = 0.0
            self.get_logger().info('Moving backward')
        elif self.gesture_state == "TURN_LEFT":
            self.cmd.linear.x = 0.0
            self.cmd.angular.z = 0.2  # Fixed turning speed
            self.get_logger().info('Turning left')

        # Publish command
        self.publisher_.publish(self.cmd)

        # Show camera feed
        cv2.imshow("Hand Gesture Control", img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    controller = HandGestureAndLidar()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.cap.release()
        cv2.destroyAllWindows()
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
