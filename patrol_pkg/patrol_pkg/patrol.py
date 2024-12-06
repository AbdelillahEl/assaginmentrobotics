import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import ReliabilityPolicy, QoSProfile

class SimplePatrol(Node):
    def __init__(self):
        super().__init__('simple_patrol')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.subscriber = self.create_subscription(
            LaserScan, 
            '/scan', 
            self.laser_callback, 
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )
        self.laser_forward = float('inf')
        self.create_timer(0.1, self.motion)
        self.cmd = Twist()

    def laser_callback(self, msg):
        self.laser_forward = msg.ranges[0]

    def motion(self):
        if self.laser_forward < 0.5:
            self.cmd.linear.x = 0.0
            self.cmd.angular.z = 0.0
            self.get_logger().info('Obstacle detected! Stopping.')
        else:
            self.cmd.linear.x = 0.2
            self.cmd.angular.z = 0.0
        self.publisher_.publish(self.cmd)

def main(args=None):
    rclpy.init(args=args)
    simple_patrol = SimplePatrol()
    rclpy.spin(simple_patrol)
    simple_patrol.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
