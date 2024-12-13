import cv2
import mediapipe as mp
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math

class HandDetectionNode(Node):
    def __init__(self):
        super().__init__('hand_detection_node')

        # ROS2 Publisher for robot movement
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Mediapipe Hand Detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils  # Import the drawing utils

        # Camera Setup
        self.window_name = "cam1"
        cv2.namedWindow(self.window_name)
        self.cap = cv2.VideoCapture("http://10.2.172.103:8080/?action=stream")

        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera stream.")
            return

        self.timer = self.create_timer(0.1, self.process_frame)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to read frame from camera.")
            return

        # Rescale frame
        frame = self.rescale_frame(frame, 75)

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands
        result = self.hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Check if all five fingers are open, or perform specific gestures
                if self.is_all_fingers_open(hand_landmarks):
                    self.get_logger().info("All fingers open! Moving forward.")
                    self.move_forward()
                elif self.is_index_finger_shown(hand_landmarks):
                    self.get_logger().info("Index finger shown! Moving backward.")
                    self.move_backward()
                elif self.is_index_and_middle_finger_shown(hand_landmarks):
                    self.get_logger().info("Index + middle finger shown! Turning around.")
                    self.turn_around()
                else:
                    self.get_logger().info("Hand is closed! Stopping robot.")
                    self.stop_robot()

                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Display the frame
        cv2.imshow(self.window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            self.destroy()

    def is_all_fingers_open(self, hand_landmarks):
        # Check if all five fingers are open by checking the distances from tips to bases
        open_threshold = 0.05  # You can adjust this threshold value

        # Thumb (landmark 4, base 2)
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_base = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]

        # Index finger (landmark 8, base 5)
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_base = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]

        # Middle finger (landmark 12, base 9)
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_base = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

        # Ring finger (landmark 16, base 13)
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_base = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]

        # Pinky (landmark 20, base 17)
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_base = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]

        # Calculate distances between tip and base of each finger
        thumb_distance = self.calculate_distance(thumb_tip, thumb_base)
        index_distance = self.calculate_distance(index_tip, index_base)
        middle_distance = self.calculate_distance(middle_tip, middle_base)
        ring_distance = self.calculate_distance(ring_tip, ring_base)
        pinky_distance = self.calculate_distance(pinky_tip, pinky_base)

        # If all distances are greater than the threshold, the hand is fully open
        if (thumb_distance > open_threshold and
            index_distance > open_threshold and
            middle_distance > open_threshold and
            ring_distance > open_threshold and
            pinky_distance > open_threshold):
            return True
        else:
            return False

    def is_index_finger_shown(self, hand_landmarks):
        # Get the position of the index finger tip (landmark 8)
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # We check if the index finger is extended by comparing its position to the base of the hand
        index_base = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        distance = self.calculate_distance(index_tip, index_base)

        # If the index finger is extended (distance > threshold), return True
        return distance > 0.05  # Adjust this threshold as needed

    def is_index_and_middle_finger_shown(self, hand_landmarks):
        # Get the positions of the index and middle finger tips (landmarks 8 and 12)
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

        # Calculate the distance between the index and middle finger tips
        distance = self.calculate_distance(index_tip, middle_tip)

        # If the index and middle fingers are both extended (distance > threshold), return True
        return distance > 0.05  # Adjust this threshold as needed

    def calculate_distance(self, point1, point2):
        # Calculate the Euclidean distance between two points (x, y, z)
        return math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2 + (point2.z - point1.z) ** 2)

    def move_forward(self):
        twist = Twist()
        twist.linear.x = 0.1  # Move forward slowly
        twist.angular.z = 0.0
        self.cmd_vel_publisher.publish(twist)

    def move_backward(self):
        twist = Twist()
        twist.linear.x = -0.1  # Move backward slowly
        twist.angular.z = 0.0
        self.cmd_vel_publisher.publish(twist)

    def turn_around(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.5  # Turn around slowly
        self.cmd_vel_publisher.publish(twist)

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_publisher.publish(twist)

    def rescale_frame(self, frame, percent):
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    def destroy(self):
        self.cap.release()
        cv2.destroyAllWindows()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    hand_detection_node = HandDetectionNode()

    try:
        rclpy.spin(hand_detection_node)
    except KeyboardInterrupt:
        hand_detection_node.get_logger().info("Shutting down hand detection node.")
        hand_detection_node.destroy()

if __name__ == '__main__':
    main()
