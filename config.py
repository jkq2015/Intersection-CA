from enum import Enum

# 仿真步长
DT = 0.1

# 环境
CANVAS_E = 400
LANE_WIDTH = 8
START_DIS = 70
left_side = CANVAS_E / 2 - 3 * LANE_WIDTH
right_side = CANVAS_E / 2 + 3 * LANE_WIDTH

# 车辆
CAR_WIDTH = 4
CAR_LEN = 4
V_MAX = 50.0/3
A_MAX = 3
A_MIN = -3
A_STATUS = -2  # status = -1时车辆以此加速度减速

# 训练参数
MAX_EPISODES = 1000
MEMORY_CAPACITY = 10000
S_DIM = 4
A_DIM = 2
VAR = 6  # control exploration


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3