import time
import tkinter as tk
import numpy as np
import config
import car
import random

class Agent:
    def __init__(self, id, plt):
        self.id = id
        self.platoons = plt             # 一个Agent里面的车队数量为2
        self.is_p0_min = -1             # 标志p0还是p1能更先到达冲突区(p0是self.platoons[0]),这在后面计算状态时会用到
        self.observation = [0, 0]
        self.reward = 0
        self.done = False               # Agent中的两个车队的控制过程是否已经结束(在cal_rsd里会有计算方式)

        min_a = min(self.platoons[0].a[0], self.platoons[1].a[0])
        self.observation, self.reward, self.done = self.cal_rsd(min_a)  # 计算reward,state,done并赋值

    # 根据强化学习计算出的action采取行动
    def step(self, action):
        next_a = [self.platoons[0].a[0], self.platoons[1].a[0]]
        for i in range(2):
            if self.platoons[i].taken_action != -1 and self.platoons[i].taken_action != self.id:
                # 如果已经采取措施并且采取的措施不是跟随着本Agent的，那么跳过
                continue
            if self.is_p0_min == 1:
                if self.platoons[i].status == -2:               # status=-2的车队被迫减速
                    self.platoons[i].a[0] = config.A_STATUS
                    next_a[i] = config.A_STATUS
                else:
                    self.platoons[i].a[0] = action[i]           # 否则按照action行动
                    next_a[i] = action[i]

            else:
                if self.platoons[i].status == -2:
                    self.platoons[i].a[0] = config.A_STATUS
                    next_a[i] = config.A_STATUS
                else:
                    self.platoons[i].a[0] = action[1 - i]
                    next_a[i] = action[1 - i]
            self.platoons[i].update()
            self.platoons[i].taken_action = self.id

        min_a = min(next_a[0], next_a[1])
        self.observation, self.reward, self.done = self.cal_rsd(min_a)      # 计算新的奖励、状态等
        return self.done

    # 计算参数reward,state,done
    def cal_rsd(self, min_a):
        p0_min_time, p0_max_time, p0_next_v, p1_min_time, p1_max_time, p1_next_v, collision_time = cal_time(self.platoons[0], self.platoons[1])

        # whether done
        if p0_max_time*p0_min_time < 0 and p1_max_time*p1_min_time < 0:     # 碰撞时done改为True
            done = True
        elif self.platoons[0].reach_des() or self.platoons[1].reach_des() \
                or (p0_next_v < 0.01 and p1_next_v < 0.01):                 # 一方到达目的线或者速度基本减为0时done也改为True
            done = True
        else:
            done = False

        # reward
        if collision_time > -1:
            reward = - 5*collision_time
        elif p0_next_v < 3 or p1_next_v < 3:
            reward = - 1.0/(p0_next_v + 0.01) - 1.0/(p1_next_v + 0.01)
        else:
            reward = 5 - 4.0 / (min_a - config.A_MAX - 0.01)

        if self.is_p0_min == -1:
            if p0_min_time < p1_min_time:
                self.is_p0_min = 1
            else:
                self.is_p0_min = 0

        # state
        if self.is_p0_min == 1:
            s_ = np.array([p0_min_time, p0_max_time, p1_min_time, p1_max_time])
        else:
            s_ = np.array([p1_min_time, p1_max_time, p0_min_time, p0_max_time])
        return s_, reward, done


class Crossing(tk.Tk, object):
    def __init__(self, platoons):
        super(Crossing, self).__init__()
        self.title('Crossing')
        self.geometry('{0}x{1}'.format(config.CANVAS_E, config.CANVAS_E))
        self.platoons = platoons
        self.platoons_show = []

        self.agents = []
        self.agent_index = []
        self.free_plts = []

        self._build_()

    def _build_(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=config.CANVAS_E,
                                width=config.CANVAS_E)

        # create lines
        x0, y0, x1, y1 = config.left_side, 0, config.left_side, config.left_side
        self.canvas.create_line(x0, y0, x1, y1)
        self.canvas.create_line(y0, x0, y1, x1)
        x0, y0, x1, y1 = config.left_side, config.right_side, config.left_side, config.CANVAS_E
        self.canvas.create_line(x0, y0, x1, y1)
        self.canvas.create_line(y0, x0, y1, x1)
        x0, y0, x1, y1 = config.right_side, 0, config.right_side, config.left_side
        self.canvas.create_line(x0, y0, x1, y1)
        self.canvas.create_line(y0, x0, y1, x1)
        x0, y0, x1, y1 = config.right_side, config.right_side, config.right_side, config.CANVAS_E
        self.canvas.create_line(x0, y0, x1, y1)
        self.canvas.create_line(y0, x0, y1, x1)

        x0, y0, x1, y1 = config.left_side, config.left_side - config.START_DIS, config.right_side, \
            config.left_side - config.START_DIS
        self.canvas.create_line(x0, y0, x1, y1, fill='red')
        self.canvas.create_line(y0, x0, y1, x1, fill='red')
        x0, y0, x1, y1 = config.left_side, config.right_side + config.START_DIS, config.right_side, \
            config.right_side + config.START_DIS
        self.canvas.create_line(x0, y0, x1, y1, fill='red')
        self.canvas.create_line(y0, x0, y1, x1, fill='red')

        x0, y0, x1, y1 = config.CANVAS_E/2, 0, config.CANVAS_E/2, config.left_side
        self.canvas.create_line(x0, y0, x1, y1, fill='orange')
        self.canvas.create_line(y0, x0, y1, x1, fill='orange')
        x0, y0, x1, y1 = config.CANVAS_E/2, config.right_side, config.CANVAS_E/2, config.CANVAS_E
        self.canvas.create_line(x0, y0, x1, y1, fill='orange')
        self.canvas.create_line(y0, x0, y1, x1, fill='orange')

        for i in range(len(self.platoons)):
            platoon_show = []
            for j in range(self.platoons[i].get_num()):
                x = self.platoons[i].x[j]
                y = self.platoons[i].y[j]
                platoon_show.append(self.canvas.create_rectangle(x, y, x + config.CAR_LEN, y + config.CAR_WIDTH))
            self.platoons_show.append(platoon_show)

        # pack all
        self.canvas.pack()

    def create(self):
        if random.random() < 0.3*config.DT*np.exp(-0.3*config.DT):
            p_new = car.Platoon(len(self.platoons), np.array([1]), np.array([config.CANVAS_E / 2 + 1.5*config.LANE_WIDTH]),
                                np.array([10]), np.array([0]), config.Direction.RIGHT, config.Direction.RIGHT)
            self.platoons.append(p_new)
            platoon_show = []
            for j in range(p_new.get_num()):
                x = p_new.x[j]
                y = p_new.y[j]
                platoon_show.append(self.canvas.create_rectangle(x, y, x + config.CAR_LEN, y + config.CAR_WIDTH))
            self.platoons_show.append(platoon_show)

    def step(self, action):
        done = True                             # 表示整个场景的仿真是否完成
        for i in range(len(self.agents)):
            di = self.agents[i].step(action[i])
            if not di:                          # 只要有一个Agent的done不是true,就认为整个场景的仿真没有完成
                done = False

        for i in range(len(self.free_plts)):
            self.free_plts[i].a[0] = 0
            # update
            self.free_plts[i].update()
            if not self.free_plts[i].reach_des():   # 只要有一个free platoon尚未到达目的地,就认为整个场景的仿真没有完成
                done = False

        # 界面上的“方形”相应移动
        for i in range(len(self.platoons)):
            for j in range(self.platoons[i].get_num()):
                coords = self.canvas.coords(self.platoons_show[i][j])
                dx_int = round(self.platoons[i].x[j] - coords[0])
                dy_int = round(self.platoons[i].y[j] - coords[1])
                self.canvas.move(self.platoons_show[i][j], dx_int, dy_int)

        return done

    def render(self):
        time.sleep(0.02)
        self.update()

    def cal_agent_free(self):                       # calculate agent and free platoons
        self.free_plts = []

        in_region_index = []                        # 指示哪些车队在控制区内
        free_index = []                             # 指示哪些车队没有加入Agent（自由行驶）

        # 删除done=True的Agents
        i = 0
        while i < len(self.agents):
            if self.agents[i].done:
                self.agents[i].platoons[0].free = True
                self.agents[i].platoons[1].free = True
                self.agents[i].platoons[0].taken_action = -1
                self.agents[i].platoons[1].taken_action = -1
                del self.agent_index[i]
                del self.agents[i]
                continue
            i = i + 1

        # 计算进入控制区域的车队
        for i in range(len(self.platoons)):
            if self.platoons[i].reach_start_line():
                in_region_index.append(i)

        for i in range(len(in_region_index)-1):
            for j in range(i+1, len(in_region_index)):
                index1 = in_region_index[i]
                index2 = in_region_index[j]
                if [index1, index2] in self.agent_index or [index2, index1] in self.agent_index:  # 如果index1和index2
                                                                                        # 已经组成了一个Agent，则continue
                    self.platoons[index1].free = False
                    self.platoons[index2].free = False
                    continue

                p0_min_time, p0_max_time, p0_next_v, p1_min_time, p1_max_time, \
                p1_next_v, collision_time = cal_time(self.platoons[index1], self.platoons[index2])

                if collision_time > 0:          # 当两车队会发生碰撞时则将他们组成一个新的Agent
                    self.agent_index.append([index1, index2])
                    new_a = Agent(len(self.agents), [self.platoons[index1], self.platoons[index2]])
                    self.agents.append(new_a)
                    self.platoons[index1].free = False
                    self.platoons[index2].free = False

        # 计算各个车队的status
        for i in range(len(self.agents)):
            p0 = self.agents[i].platoons[0]
            p1 = self.agents[i].platoons[1]
            p0_min_time, p0_max_time, p0_next_v, p1_min_time, p1_max_time, \
            p1_next_v, collision_time = cal_time(p0, p1)

            if p0.status == 0 and p1.status == 0:
                p0.status = np.sign(p1_min_time - p0_min_time)
                p1.status = np.sign(p0_min_time - p1_min_time)
            elif p0.status == 0 and p1.status != 0:
                if (p0_min_time - p1_min_time) * p1.status > 0:  # 如果此条件满足，意味着p1先到达冲突区域并且正在采取加速，或者
                    # 后到达冲突区域并且正在减速，此时p0和p1可以按照正常强化学习策
                    # 略进行控制，所以p0.status设为1
                    p0.status = np.sign(p1_min_time - p0_min_time)
                else:  # 否则意味着p0和p1如果按照强化学习策略进行控制，会和其他Agent
                    # 冲突，所以此时p0被限制只能减速，status设为-1
                    p0.status = -2

                # else:                                               # 若a1 == 0说明同时增加了两个agents，需要考虑上一个的情况
                #     p2 = self.agents[i - 1].platoons[0]
                #     p1_last = self.agents[i - 1].platoons[1]
                #     if p1_last.id != p1.id:
                #         temp = p2
                #         p2 = p1_last
                #         p1_last = temp
                #     p2_min_time, p2_max_time, p2_next_v, p1_l_min_time, p1_l_max_time, \
                #     p1_l_next_v, collision_time_last = cal_time(p2, p1_last)
                #
                #     if (p0_min_time - p1_min_time) * (p2_min_time - p1_l_min_time) > 0:
                #                                                     # 如果此条件满足，意味着p1先到达冲突区域并且正在采取加速，或者
                #                                                     # 后到达冲突区域并且正在减速，此时p0和p1可以按照正常强化学习策
                #                                                     # 略进行控制，所以p0.status设为1
                #         p0.status = 1
                #     else:                                           # 否则意味着p0和p1如果按照强化学习策略进行控制，会和其他Agent
                #                                                     # 冲突，所以此时p0被限制只能减速，status设为-1
                #         p0.status = -1

            elif p0.status != 0 and p1.status == 0:
                if (p1_min_time - p0_min_time) * p0.status > 0:
                    p1.status = np.sign(p0_min_time - p1_min_time)
                else:
                    p1.status = -2

                # else:
                #     p0_last = self.agents[i - 1].platoons[0]
                #     p2 = self.agents[i - 1].platoons[1]
                #     if p0_last.id != p0.id:
                #         temp = p0_last
                #         p0_last = p2
                #         p2 = temp
                #     p0_l_min_time, p0_l_max_time, p0_l_next_v, p2_min_time, p2_max_time, \
                #     p2_next_v, collision_time_last = cal_time(p0_last, p2)
                #
                #     if (p1_min_time - p0_min_time) * (p2_min_time - p0_l_min_time) > 0:
                #         p1.status = 1
                #     else:
                #         p1.status = -1

            elif p0.status == -2:
                if p1_min_time < 0:                             # 当某一个车队status为-2时，如果对方已经到达冲突区域，认为已
                                                                # 经比较安全，可以取消限制
                    p0.status = 1
            elif p1.status == -2:
                if p0_min_time < 0:
                    p1.status = 1

        # 计算free platoons
        for i in range(len(self.platoons)):
            if self.platoons[i].free:
                free_index.append(i)
                self.free_plts.append(self.platoons[i])

        print(self.agent_index)
        print(free_index)
        for i in range(len(self.platoons)):
            print(self.platoons[i].status, self.platoons[i].taken_action)
        print('**************')


# calculate time for straight and straight
def cal_time_ss(p0, p1):
    if p0.start_dir == p1.start_dir or \
            (p0.start_dir == config.Direction.LEFT and p1.start_dir == config.Direction.RIGHT) or \
            (p1.start_dir == config.Direction.LEFT and p0.start_dir == config.Direction.RIGHT) or \
            (p0.start_dir == config.Direction.UP and p1.start_dir == config.Direction.DOWN) or \
            (p1.start_dir == config.Direction.UP and p0.start_dir == config.Direction.DOWN):   # 不会相撞
        return 1,1,10,1,1,10,-1

    # 下面的p_lr和p_ud分别代表左右行驶和上下行驶的车队，左右和上下的信息在下面的计算中要用
    if p0.start_dir == config.Direction.RIGHT or p0.start_dir == config.Direction.LEFT:
        p_lr = p0
        p_ud = p1
    else:
        p_lr = p1
        p_ud = p0

    lr_leader_co = [p_lr.x[0], p_lr.y[0]]  # left_right
    lr_tail_co = [p_lr.x[-1], p_lr.y[-1]]
    lr_next_v = p_lr.v[0]

    ud_leader_co = [p_ud.x[0], p_ud.y[0]]  # up_down
    ud_tail_co = [p_ud.x[-1], p_ud.y[-1]]
    ud_next_v = p_ud.v[0]

    # collision location
    col = [ud_leader_co[0], lr_leader_co[1]]

    if p_lr.start_dir == config.Direction.RIGHT:
        lr_flag = 1
    else:
        lr_flag = -1

    if p_ud.start_dir == config.Direction.DOWN:
        ud_flag = 1
    else:
        ud_flag = -1

    # calculate reward
    min_d_lr = lr_flag * (col[0] - lr_leader_co[0]) - config.CAR_LEN # min distance (to collision area) of p_lr, 即p_lr的头车车头到冲突区域对应边缘的距离
                                                    # 下面的变量类似
    max_d_lr = lr_flag * (col[0] - lr_tail_co[0]) + config.CAR_LEN
    min_d_ud = ud_flag * (col[1] - ud_leader_co[1]) - config.CAR_LEN
    max_d_ud = ud_flag * (col[1] - ud_tail_co[1]) + config.CAR_LEN

    lr_min_time = min_d_lr / (lr_next_v + 0.1)      # min time (to collision area) of p_lr, p_lr到达冲突区的最短时间，下面的变量类似
    lr_min_time = min(lr_min_time, 40)
    lr_max_time = max_d_lr / (lr_next_v + 0.1)
    lr_max_time = min(lr_max_time, 40)
    ud_min_time = min_d_ud / (ud_next_v + 0.1)
    ud_min_time = min(ud_min_time, 40)
    ud_max_time = max_d_ud / (ud_next_v + 0.1)
    ud_max_time = min(ud_max_time, 40)

    collision_time = min(lr_max_time - ud_min_time, ud_max_time - lr_min_time)
    if p0.start_dir == config.Direction.RIGHT or p0.start_dir == config.Direction.LEFT:
        # print(lr_min_time, lr_max_time, lr_next_v, ud_min_time, ud_max_time)
        return lr_min_time, lr_max_time, lr_next_v, ud_min_time, ud_max_time, ud_next_v, collision_time
    else:
        # print(ud_min_time, ud_max_time, ud_next_v, lr_min_time, lr_max_time)
        return ud_min_time, ud_max_time, ud_next_v, lr_min_time, lr_max_time, lr_next_v, collision_time


# calculate time for straight and turn
def cal_time_st(p0, p1):
    if p0.start_dir == p1.start_dir:
        return 1, 1, 10, 1, 1, 10, -1
    # p_s代表直行车，p_r代表转弯车
    if p0.start_dir == p0.end_dir:
        p_s = p0
        p_t = p1
    else:
        p_s = p1
        p_t = p0

    s_leader_co = [p_s.x[0], p_s.y[0]]
    s_tail_co = [p_s.x[-1], p_s.y[-1]]
    s_next_v = p_s.v[0]

    t_leader_co = [p_t.x[0], p_t.y[0]]
    t_tail_co = [p_t.x[-1], p_t.y[-1]]
    t_next_v = p_t.v[0]

    # 计算冲突区的x,y坐标（col_x和col_y）
    col_x = 0
    col_y = 0
    if p_s.start_dir == config.Direction.RIGHT:
        if p_t.start_dir == config.Direction.UP and p_t.end_dir == config.Direction.LEFT:
            col_y = p_s.y[0]
            col_x = config.left_side + np.sqrt(p_t.radius**2 - (config.right_side - col_y)**2)
        elif p_t.start_dir == config.Direction.LEFT and p_t.end_dir == config.Direction.DOWN:
            col_y = p_s.y[0]
            col_x = config.right_side - np.sqrt(p_t.radius**2 - (config.right_side - col_y)**2)
        elif (p_t.start_dir == config.Direction.UP and p_t.end_dir == config.Direction.RIGHT)\
                or (p_t.start_dir == config.Direction.DOWN and p_t.end_dir == config.Direction.RIGHT):
            col_y = p_s.y[0]
            col_x = config.right_side
    elif p_s.start_dir == config.Direction.LEFT:
        if p_t.start_dir == config.Direction.DOWN and p_t.end_dir == config.Direction.RIGHT:
            col_y = p_s.y[0]
            col_x = config.right_side - np.sqrt(p_t.radius**2 - (config.left_side - col_y)**2)
        elif p_t.start_dir == config.Direction.RIGHT and p_t.end_dir == config.Direction.UP:
            col_y = p_s.y[0]
            col_x = config.left_side + np.sqrt(p_t.radius**2 - (config.left_side - col_y)**2)
        elif (p_t.start_dir == config.Direction.UP and p_t.end_dir == config.Direction.LEFT) \
                 or (p_t.start_dir == config.Direction.DOWN and p_t.end_dir == config.Direction.LEFT):
            col_y = p_s.y[0]
            col_x = config.left_side
    elif p_s.start_dir == config.Direction.UP:
        if p_t.start_dir == config.Direction.DOWN and p_t.end_dir == config.Direction.RIGHT:
            col_x = p_s.x[0]
            col_y = config.left_side + np.sqrt(p_t.radius**2 - (config.right_side - col_x)**2)
        elif p_t.start_dir == config.Direction.LEFT and p_t.end_dir == config.Direction.DOWN:
            col_x = p_s.x[0]
            col_y = config.right_side - np.sqrt(p_t.radius**2 - (config.right_side - col_x)**2)
        elif (p_t.start_dir == config.Direction.RIGHT and p_t.end_dir == config.Direction.UP)\
                or (p_t.start_dir == config.Direction.LEFT and p_t.end_dir == config.Direction.UP):
            col_x = p_s.x[0]
            col_y = config.left_side
    else:
        if p_t.start_dir == config.Direction.RIGHT and p_t.end_dir == config.Direction.UP:
            col_x = p_s.x[0]
            col_y = config.left_side + np.sqrt(p_t.radius**2 - (config.left_side - col_x)**2)
        elif p_t.start_dir == config.Direction.UP and p_t.end_dir == config.Direction.LEFT:
            col_x = p_s.x[0]
            col_y = config.right_side - np.sqrt(p_t.radius**2 - (config.left_side - col_x)**2)
        elif (p_t.start_dir == config.Direction.RIGHT and p_t.end_dir == config.Direction.DOWN)\
                or (p_t.start_dir == config.Direction.LEFT and p_t.end_dir == config.Direction.DOWN):
            col_x = p_s.x[0]
            col_y = config.right_side

    # collision location
    col = [col_x, col_y]
    if col == [0, 0]:   # 如果到这里col_x和col_y仍然为0，意味着两车队没有冲突
        return 1, 1, 10, 1, 1, 10, -1

    # calculate time of p_s
    if p_s.start_dir == config.Direction.RIGHT:
        col_index = 0  # index of 'col' to calculate time
        s_flag = 1
    elif p_s.start_dir == config.Direction.LEFT:
        col_index = 0
        s_flag = -1
    elif p_s.start_dir == config.Direction.DOWN:
        col_index = 1
        s_flag = 1
    else:
        col_index = 1
        s_flag = -1

    min_d_s = s_flag * (col[col_index] - s_leader_co[col_index])
    max_d_s = s_flag * (col[col_index] - s_tail_co[col_index])
    s_min_time = min_d_s / (s_next_v + 0.1)
    s_min_time = min(s_min_time, 40)
    s_max_time = max_d_s / (s_next_v + 0.1)
    s_max_time = min(s_max_time, 40)

    # calculate time of p_t
    theta_col = np.arctan(abs((p_t.center[1] - col[1])/(p_t.center[0] - col[0])))
    theta_leader = np.arctan(abs((p_t.center[1] - t_leader_co[1])/(p_t.center[0] - t_leader_co[0])))
    theta_tail = np.arctan(abs((p_t.center[1] - t_tail_co[1])/(p_t.center[0] - t_tail_co[0])))

    if not car.arrive_crossing(t_leader_co[0], t_leader_co[1], p_t.start_dir):
        if p_t.start_dir == config.Direction.UP:
            min_d_t = t_leader_co[1] - config.right_side + theta_col * p_t.radius
        elif p_t.start_dir == config.Direction.DOWN:
            min_d_t = config.left_side - t_leader_co[1] + theta_col * p_t.radius
        elif p_t.start_dir == config.Direction.RIGHT:
            min_d_t = config.left_side - t_leader_co[0] + (np.pi/2 - theta_col) * p_t.radius
        else:
            min_d_t = t_leader_co[0] - config.right_side + (np.pi/2 - theta_col) * p_t.radius
    elif car.leave_crossing(t_leader_co[0],t_leader_co[1],p_t.end_dir):
        if p_t.end_dir == config.Direction.UP:
            min_d_t = t_leader_co[1] - config.left_side - theta_col*p_t.radius
        elif p_t.end_dir == config.Direction.DOWN:
            min_d_t = config.right_side - t_leader_co[1] - theta_col*p_t.radius
        elif p_t.end_dir == config.Direction.RIGHT:
            min_d_t = config.right_side - t_leader_co[0] - (np.pi/2 - theta_col) * p_t.radius
        else:
            min_d_t = t_leader_co[0] - config.left_side - (np.pi/2 - theta_col) * p_t.radius
    else:
        if p_t.start_dir == config.Direction.UP or p_t.start_dir == config.Direction.DOWN:
            min_d_t = (theta_col - theta_leader)*p_t.radius
        else:
            min_d_t = (theta_leader - theta_col)*p_t.radius

    if not car.arrive_crossing(t_tail_co[0], t_tail_co[1], p_t.start_dir):
        if p_t.start_dir == config.Direction.UP:
            max_d_t = t_tail_co[1] - config.right_side + theta_col*p_t.radius
        elif p_t.start_dir == config.Direction.DOWN:
            max_d_t = config.left_side - t_tail_co[1] + theta_col*p_t.radius
        elif p_t.start_dir == config.Direction.RIGHT:
            max_d_t = config.left_side - t_tail_co[0] + (np.pi/2 - theta_col) * p_t.radius
        else:
            max_d_t = t_tail_co[0] - config.right_side + (np.pi/2 - theta_col) * p_t.radius
    elif car.leave_crossing(t_tail_co[0],t_tail_co[1],p_t.end_dir):
        if p_t.end_dir == config.Direction.UP:
            max_d_t = t_tail_co[1] - config.left_side - theta_col*p_t.radius
        elif p_t.end_dir == config.Direction.DOWN:
            max_d_t = config.right_side - t_tail_co[1] - theta_col*p_t.radius
        elif p_t.end_dir == config.Direction.RIGHT:
            max_d_t = config.right_side - t_tail_co[0] - (np.pi/2 - theta_col) * p_t.radius
        else:
            max_d_t = t_tail_co[0] - config.left_side - (np.pi/2 - theta_col) * p_t.radius
    else:
        if p_t.start_dir == config.Direction.UP or p_t.start_dir == config.Direction.DOWN:
            max_d_t = (theta_col - theta_tail)*p_t.radius
        else:
            max_d_t = (theta_tail - theta_col)*p_t.radius

    t_min_time = min_d_t / (t_next_v + 0.1)
    t_min_time = min(t_min_time, 40)
    t_max_time = max_d_t / (t_next_v + 0.1)
    t_max_time = min(t_max_time, 40)
    # print(s_min_time, s_max_time, t_min_time, t_max_time)

    collision_time = min(s_max_time - t_min_time, t_max_time - s_min_time)
    if p0.start_dir == p0.end_dir:
        return s_min_time, s_max_time, s_next_v, t_min_time, t_max_time, t_next_v, collision_time
    else:
        return t_min_time, t_max_time, t_next_v, s_min_time, s_max_time, s_next_v, collision_time


# 计算各个时间，目前只考虑了两队都直行，一车直行一车转弯的情况，尚未考虑两车都转弯的情况
def cal_time(p0,p1):
    if p0.start_dir == p0.end_dir and p1.start_dir == p1.end_dir:
        return cal_time_ss(p0, p1)
    elif p0.start_dir != p0.end_dir and p1.start_dir != p1.end_dir:
        return 0
    else:
        return cal_time_st(p0,p1)
