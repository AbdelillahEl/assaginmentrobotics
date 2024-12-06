import rclpy
# import the ROS2 python libraries
from rclpy.node import Node
# import the Twist module from geometry_msgs interface
from geometry_msgs.msg import Twist
# import the LaserScan module from sensor_msgs interface
from sensor_msgs.msg import LaserScan
from rclpy.qos import ReliabilityPolicy, QoSProfile

class Patrol(Node):

    def __init__(self):
        # Here you have the class constructor
        # call the class constructor
        super().__init__('patrol')
        # create the publisher object
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        # create the subscriber object
        self.subscriber = self.create_subscription(
            LaserScan, '/scan', self.laser_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )
        # define the timer period for 0.5 seconds
        self.timer_period = 0.5
        # define the variable to save the received info
        self.laser_forward = 0
        self.laser_frontLeft = 0
        self.laser_frontRight = 0

        # create a Twist message
        self.cmd = Twist()
        self.timer = self.create_timer(self.timer_period, self.motion)

    def laser_callback(self, msg):
        # Save the frontal laser scan info at 0°
        self.laser_forward = msg.ranges[359]
        self.laser_frontLeft = min(msg.ranges[0:15])
        self.laser_frontRight = min(msg.ranges[345:359])

    def motion(self):
        # Print the data
        self.get_logger().info('Forward: "%s"' % str(self.laser_forward))
        
        # Logic for avoiding obstacles
        if self.laser_frontLeft < 0.5:
            self.get_logger().info('Object detected on front left: "%s"' % str(self.laser_frontLeft))
            # Turn right
            self.cmd.linear.x = 0.1  # Adjust forward speed as necessary
            self.cmd.angular.z = -0.5  # Negative for right turn
        elif self.laser_frontRight < 0.5:
            self.get_logger().info('Object detected on front right: "%s"' % str(self.laser_frontRight))
            # Turn left
            self.cmd.linear.x = 0.1
            self.cmd.angular.z = 0.5  # Positive for left turn
        else:
            # Move forward if no obstacle is close
            self.cmd.linear.x = 0.2  # Forward speed
            self.cmd.angular.z = 0.0  # No rotation

        # Publishing the cmd_vel values to a Topic
        self.publisher_.publish(self.cmd)

def main(args=None):
    # initialize the ROS communication
    rclpy.init(args=args)
    # declare the node constructor
    patrol = Patrol()       
    # pause the program execution, waits for a request to kill the node (ctrl+c)
    rclpy.spin(patrol)
    # Explicitly destroy the node
    patrol.destroy_node()
    # shutdown the ROS communication
    rclpy.shutdown()

if __name__ == '__main__':
    main()
