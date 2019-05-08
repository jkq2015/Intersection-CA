from enum import Enum

# 仿真步长
DT = 0.1

# 环境
CANVAS_E = 600
LANE_WIDTH = 8
left_side = CANVAS_E / 2 - 3 * LANE_WIDTH
right_side = CANVAS_E / 2 + 3 * LANE_WIDTH
SAFE_DIS = 20
SAFE_DIS_CACC = 12

# 车辆
CAR_WIDTH = 4
CAR_LEN = 4
V_MAX = 50.0/3
A_MAX = 3
A_MIN = -3
A_STATUS = -2  # status = -2时车辆以此加速度减速
MAX_PLATOON_SIZE = 5

# 训练参数
MAX_EPISODES = 1000
MEMORY_CAPACITY = 10000
S_DIM = 4
A_DIM = 2
VAR = 6  # control exploration

# 策略


class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
