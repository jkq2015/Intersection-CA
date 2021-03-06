import numpy as np
import config


class Platoon:
    def __init__(self, p_id, p_init_x, p_init_y, p_init_v, p_init_a, p_start_dir, p_end_dir):
        self.id = p_id
        self.x = p_init_x
        self.y = p_init_y
        self.v = p_init_v
        self.a = p_init_a

        self.theta = np.zeros(len(self.x))             # 转角
        self.start_dir = p_start_dir
        self.end_dir = p_end_dir
        self.taken_action = -1        # 指示该车队的决策依照的Agent,取值为依照的Agent的id，如果尚未确定依照哪个Agent,则取-1
        self.center, self.radius = cal_cr(self.start_dir, self.end_dir, self.x[0], self.y[0])

        self.status = 0  # 指示车队状态，如果车队跟着某个Agent进行决策则取1或-1，分别对应加速、减速的情况，如果被限制只能减速则取-2，如果尚未确定状态0
        self.constrain_p = -1       # 指示当status为-2时限制该车队的车队id
        self.free = True
        self.time = np.zeros(len(self.x))
        self.whether_pass_crossing = np.zeros(len(self.x), int)

        for i in range(len(self.x)):
            self.theta[i] = cal_theta(self.start_dir, self.end_dir, self.x[i], self.y[i])

    def get_num(self):
        return len(self.x)

    def copy_from(self, p):
        self.id = p.id
        self.x = p.x
        self.y = p.y
        self.v = p.v
        self.a = p.a
        self.theta = p.theta
        self.start_dir = p.start_dir
        self.end_dir = p.end_dir
        self.taken_action = p.taken_action
        self.center = p.center
        self.radius = p.radius
        self.status = p.status
        self.free = p.free

    def copy(self):
        p_copy = Platoon(self.id, self.x, self.y, self.v, self.a, self.start_dir, self.end_dir)
        p_copy.taken_action = self.taken_action
        p_copy.status = self.status
        p_copy.free = self.free
        return p_copy

    def add_one_car(self, x, y, v, a):
        self.x = np.hstack((self.x, x))
        self.y = np.hstack((self.y, y))
        self.v = np.hstack((self.v, v))
        self.a = np.hstack((self.a, a))
        self.theta = np.hstack((self.theta, cal_theta(self.start_dir, self.end_dir, x, y)))
        self.time = np.hstack((self.time, 0))
        self.whether_pass_crossing = np.hstack((self.whether_pass_crossing, 0))

    def follow_front_platoon(self, front_platoon):
        if front_platoon != -1:
            delta_x = np.sqrt(np.power((self.x[0] - front_platoon.x[-1]), 2) +
                              np.power((self.y[0] - front_platoon.y[-1]), 2))
            if delta_x < config.START_DIS:
                self.a[0] = follow_car_acc(delta_x, self.v[0], front_platoon.v[-1], front_platoon.a[-1])
            elif self.v[0] < config.V_MAX:
                self.a[0] = 3
            elif self.v[0] > config.V_MAX:
                self.a[0] = 0
        elif self.v[0] < config.V_MAX:
            self.a[0] = 3
        elif self.v[0] > config.V_MAX:
            self.a[0] = 0

    def add_time(self):
        for i in range(len(self.time)):
            self.time[i] += int(not reach_des_(self.x[i], self.y[i], self.end_dir))

    def cal_whether_pass(self):
        old_sum = np.sum(self.whether_pass_crossing)
        for i in range(len(self.whether_pass_crossing)):
            if leave_crossing(self.x[i], self.y[i], self.end_dir):
                self.whether_pass_crossing[i] = 1
        if np.sum(self.whether_pass_crossing) > old_sum:
            return 1
        else:
            return 0

    # 计算是否抵达目的地，目的地是地图边缘，尾车离开时算到达，返回True
    def reach_des(self):
        return reach_des_(self.x[-1], self.y[-1], self.end_dir)

    def leave_region(self):
        return leave_crossing(self.x[-1], self.y[-1], self.end_dir)

    # 计算是否进入决策区域，决策区域的起始线也是图中的红线，头车到达时算到达，返回True
    def reach_start_line(self):
        if self.start_dir == config.Direction.RIGHT:
            result = self.x[0] > config.left_side - config.START_DIS
        elif self.start_dir == config.Direction.LEFT:
            result = self.x[0] + config.CAR_LEN < config.right_side + config.START_DIS
        elif self.start_dir == config.Direction.UP:
            result = self.y[0] + config.CAR_LEN < config.right_side + config.START_DIS
        else:
            result = self.y[0] > config.left_side - config.START_DIS
        return result

    # 更新运动学状态，注意：调用此函数之前头车的加速度已经计算出，根据头车加速度和跟驰模型计算整个队伍的状态
    def update(self):
        # 计算头车转角
        self.theta[0] = cal_theta(self.start_dir, self.end_dir, self.x[0], self.y[0])

        # 计算头车后面的车的加速度
        for i in range(1, len(self.x)):
            delta_x = np.sqrt(np.power((self.x[i] - self.x[i-1]), 2) + np.power((self.y[i] - self.y[i-1]), 2))
            self.a[i] = follow_car_cacc(delta_x, self.v[i], self.v[i - 1], self.a[i - 1], self.v[0], self.a[0])
            self.theta[i] = cal_theta(self.start_dir, self.end_dir, self.x[i], self.y[i])

        # 计算速度和位置
        ds = self.v * config.DT + 0.5 * self.a * config.DT**2
        for i in range(len(self.v)):
            if self.v[i] >= config.V_MAX:
                ds[i] = config.V_MAX * config.DT

        self.v = self.v + self.a * config.DT
        self.v = np.clip(self.v, 0, config.V_MAX)

        dx = ds*np.cos(self.theta)
        dy = ds*np.sin(self.theta)
        self.x = self.x + dx
        self.y = self.y + dy


# 计算是否到达路口（和是否到达控制区域不同，控制区域是四条红线包围的区域，比路口要大）
def arrive_crossing(x, y, start_dir):
    if start_dir == config.Direction.RIGHT:
        result = x > config.left_side
    elif start_dir == config.Direction.LEFT:
        result = x + config.CAR_LEN < config.right_side
    elif start_dir == config.Direction.UP:
        result = y + config.CAR_LEN < config.right_side
    else:
        result = y > config.left_side
    return result


# 计算是否离开路口
def leave_crossing(x, y, end_dir):
    if end_dir == config.Direction.RIGHT:
        result = x > config.right_side
    elif end_dir == config.Direction.LEFT:
        result = x + config.CAR_LEN < config.left_side
    elif end_dir == config.Direction.UP:
        result = y + config.CAR_LEN < config.left_side
    else:
        result = y > config.right_side
    return result


def reach_des_(x, y, end_dir):
    if end_dir == config.Direction.RIGHT:
        result = x > config.CANVAS_E
    elif end_dir == config.Direction.LEFT:
        result = x + config.CAR_LEN < 0
    elif end_dir == config.Direction.UP:
        result = y + config.CAR_LEN < 0
    else:
        result = y > config.CANVAS_E
    return result


# 计算转弯圆心和半径（只能在到达路口前调用）
def cal_cr(start_dir, end_dir, x, y):
    center = [0, 0]
    if start_dir == end_dir:
        radius = -1
    else:
        if end_dir == config.Direction.RIGHT:
            radius = config.right_side - x
            center[0] = config.right_side
            if start_dir == config.Direction.UP:
                center[1] = config.right_side
            else:
                center[1] = config.left_side
        elif end_dir == config.Direction.LEFT:
            radius = x - config.left_side
            center[0] = config.left_side
            if start_dir == config.Direction.UP:
                center[1] = config.right_side
            else:
                center[1] = config.left_side
        elif end_dir == config.Direction.DOWN:
            radius = config.right_side - y
            center[1] = config.right_side
            if start_dir == config.Direction.LEFT:
                center[0] = config.right_side
            else:
                center[0] = config.left_side
        else:
            radius = y - config.left_side
            center[1] = config.left_side
            if start_dir == config.Direction.LEFT:
                center[0] = config.right_side
            else:
                center[0] = config.left_side
    return center, radius


# 根据车辆所处的位置计算转角
def cal_theta(start_dir, end_dir, x, y):
    if start_dir == end_dir or not arrive_crossing(x, y, start_dir):
        if start_dir == config.Direction.RIGHT:
            theta = 0
        elif start_dir == config.Direction.DOWN:
            theta = np.pi/2
        elif start_dir == config.Direction.LEFT:
            theta = np.pi
        else:
            theta = -np.pi/2
    elif leave_crossing(x, y, end_dir):
        if end_dir == config.Direction.RIGHT:
            theta = 0
        elif end_dir == config.Direction.DOWN:
            theta = np.pi/2
        elif end_dir == config.Direction.LEFT:
            theta = np.pi
        else:
            theta = -np.pi/2
    else:
        if start_dir == config.Direction.RIGHT and end_dir == config.Direction.DOWN:
            dx = config.left_side - x
            dy = config.right_side - y
            theta = np.arctan2(-dx, dy)
        elif start_dir == config.Direction.UP and end_dir == config.Direction.LEFT:
            dx = config.left_side - x
            dy = config.right_side - y
            theta = np.arctan2(dx, -dy)
        elif start_dir == config.Direction.LEFT and end_dir == config.Direction.DOWN:
            dx = config.right_side - x
            dy = config.right_side - y
            theta = np.arctan2(dx, -dy)
        elif start_dir == config.Direction.UP and end_dir == config.Direction.RIGHT:
            dx = config.right_side - x
            dy = config.right_side - y
            theta = np.arctan2(-dx, dy)
        elif start_dir == config.Direction.RIGHT and end_dir == config.Direction.UP:
            dx = config.left_side - x
            dy = config.left_side - y
            theta = np.arctan2(dx, -dy)
        elif start_dir == config.Direction.DOWN and end_dir == config.Direction.LEFT:
            dx = config.left_side - x
            dy = config.left_side - y
            theta = np.arctan2(-dx, dy)
        elif start_dir == config.Direction.LEFT and end_dir == config.Direction.UP:
            dx = config.right_side - x
            dy = config.left_side - y
            theta = np.arctan2(-dx, dy)
        else:
            dx = config.right_side - x
            dy = config.left_side - y
            theta = np.arctan2(dx, -dy)
    return theta


def follow_car_cacc(delta_x, self_v, front_v, front_a, leader_v, leader_a):
    # CACC参数
    c_1 = 0.5
    w_n = 0.2
    xi = 1
    gap = 12

    alpha_1 = 1 - c_1
    alpha_2 = c_1
    alpha_3 = -(2 * xi - c_1 * (xi + (xi * xi - 1)**0.5)) * w_n
    alpha_4 = - c_1 * (xi + (xi * xi - 1)**0.5) * w_n
    alpha_5 = -w_n * w_n

    pre_acc = front_a
    leader_acc = leader_a
    epsilon_i = -delta_x + config.CAR_LEN + gap
    d_epsilon_i = self_v - front_v
    tem_a = alpha_1 * pre_acc + alpha_2 * leader_acc + alpha_3 * d_epsilon_i + alpha_4 * (self_v - leader_v)\
        + alpha_5 * epsilon_i

    # 最大最小速度限制
    if tem_a > config.A_MAX:
        a_new = config.A_MAX
    elif tem_a < config.A_MIN:
        a_new = config.A_MIN
    else:
        a_new = tem_a

    # 发动机参数限制
    if a_new > 0:
        a_new = min(a_new, engine_speedup_acc_curve(self_v))
    else:
        a_new = max(a_new, engine_slowdown_acc_curve(self_v))
    return a_new


def follow_car_acc(delta_x, self_v, front_v, front_a):
    v1 = self_v
    v2 = front_v + front_a * config.DT
    # ACC参数
    lambda_ = 0.1
    epsilon = v1 - v2
    # ACC计算公式
    t = config.SAFE_DIS / (v1 + 0.1)
    delta = -delta_x + config.CAR_LEN + t * v1              # consider car's length and safe headway
    tem_a = -(epsilon + lambda_ * delta) / t
    # 最大最小速度限制
    if tem_a > config.A_MAX:
        a_new = config.A_MAX
    elif tem_a < config.A_MIN:
        a_new = config.A_MIN
    else:
        a_new = tem_a

    # 发动机参数限制
    if a_new > 0:
        a_new = min(a_new, engine_speedup_acc_curve(self_v))
    else:
        a_new = max(a_new, engine_slowdown_acc_curve(self_v))
    return a_new


# 对加速度的平滑函数
# 加速阶段
def engine_speedup_acc_curve(now_speed_in):
    acc_max = config.A_MAX
    v_max = config.V_MAX
    p = 0.3
    m = (v_max*p - v_max + v_max * (1 - p)**0.5) / p
    k = (1 - p)*acc_max / m / m
    engine_acc = -k*(now_speed_in - m)*(now_speed_in - m) + acc_max
    return engine_acc


# 减速阶段
def engine_slowdown_acc_curve(now_speed_in):
    acc_min = config.A_MIN
    v_max = config.V_MAX
    p = 0.6
    m = v_max / (p**0.5 + 1)
    k = -acc_min / m / m
    engine_acc = k*np.dot(now_speed_in - m, now_speed_in - m) + acc_min
    return engine_acc


def constrain_acc(a, v):
    if a > 0:
        a_new = min(a, engine_speedup_acc_curve(v))
    else:
        a_new = max(a, engine_slowdown_acc_curve(v))
    return a_new
