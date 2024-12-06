import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/abdelillah/ros2_ws/src/patrol_pkg/install/patrol_pkg'
