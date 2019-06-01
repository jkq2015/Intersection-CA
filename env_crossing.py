import time
import tkinter as tk
import numpy as np
import config
import car
import signal
import random


class Crossing(tk.Tk, object):
    def __init__(self, platoons):
        super(Crossing, self).__init__()
        self.title('Crossing')
        self.geometry('{0}x{1}'.format(config.CANVAS_E, config.CANVAS_E))
        self.signal = signal.Signal()
        self.signal_show = []
        self.platoons = platoons
        self.platoons_show = []

        self.count = 0
        self.passed_crossing = 0

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

        fill_color = ['green', 'yellow', 'red']
        offset = [1, 2, 0]
        temp = []
        for i in range(3):
            start_dir = config.Direction(0)
            lane = i
            if i == 2:
                lane = 3
            end_dir = config.Direction((0 + lane) % 4)
            temp.append(self.canvas.create_line(config.CANVAS_E / 2 + offset[i] * config.LANE_WIDTH, config.right_side,
                                                config.CANVAS_E / 2 + (offset[i] + 1) * config.LANE_WIDTH,
                                                config.right_side, fill=fill_color[self.signal.get_color(start_dir, end_dir)]))
        self.signal_show.append(temp)
        temp = []
        for i in range(3):
            start_dir = config.Direction(1)
            lane = i
            if i == 2:
                lane = 3
            end_dir = config.Direction((1 + lane) % 4)
            temp.append(self.canvas.create_line(config.left_side, config.CANVAS_E / 2 + offset[i] * config.LANE_WIDTH,
                                                config.left_side,
                                                config.CANVAS_E / 2 + (offset[i] + 1) * config.LANE_WIDTH,
                                                fill=fill_color[self.signal.get_color(start_dir, end_dir)]))
        self.signal_show.append(temp)
        temp = []
        for i in range(3):
            start_dir = config.Direction(2)
            lane = i
            if i == 2:
                lane = 3
            end_dir = config.Direction((2 + lane) % 4)
            temp.append(self.canvas.create_line(config.CANVAS_E / 2 - offset[i] * config.LANE_WIDTH, config.left_side,
                                                config.CANVAS_E / 2 - (offset[i] + 1) * config.LANE_WIDTH,
                                                config.left_side, fill=fill_color[self.signal.get_color(start_dir, end_dir)]))
        self.signal_show.append(temp)
        temp = []
        for i in range(3):
            start_dir = config.Direction(3)
            lane = i
            if i == 2:
                lane = 3
            end_dir = config.Direction((3 + lane) % 4)
            temp.append(self.canvas.create_line(config.right_side, config.CANVAS_E / 2 - offset[i] * config.LANE_WIDTH,
                                                config.right_side,
                                                config.CANVAS_E / 2 - (offset[i] + 1) * config.LANE_WIDTH,
                                                fill=fill_color[self.signal.get_color(start_dir, end_dir)]))
        self.signal_show.append(temp)

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

    def get_signal_time_left(self):
        return self.signal.get_time_left()

    def create_one_car(self, start_dir, end_dir):
        index = self.cars_in_dir(start_dir, end_dir)
        lane = (end_dir.value - start_dir.value)%4
        if lane == 3:
            lane = -1
        init_xs = [config.CANVAS_E / 2 + (lane + 1) * config.LANE_WIDTH + 2, 1,
                   config.CANVAS_E / 2 - (lane + 1) * config.LANE_WIDTH - 2 - config.CAR_LEN, config.CANVAS_E - 1]
        init_ys = [config.CANVAS_E - 1, config.CANVAS_E / 2 + (lane + 1) * config.LANE_WIDTH + 2, 1,
                   config.CANVAS_E / 2 - (lane + 1) * config.LANE_WIDTH - 2 - config.CAR_LEN]
        init_a = 0.
        init_v = config.V_MAX

        if len(index) == 0:
            create_new_platoon = True
        elif start_dir == config.Direction.NORTH:
            create_new_platoon = (self.platoons[index[-1]].y[-1] < config.CANVAS_E - config.SAFE_DIS) \
                                 or len(self.platoons[index[-1]].x) >= config.MAX_PLATOON_SIZE

        elif start_dir == config.Direction.SOUTH:
            create_new_platoon = self.platoons[index[-1]].y[-1] > config.SAFE_DIS \
                                 or len(self.platoons[index[-1]].x) >= config.MAX_PLATOON_SIZE
        elif start_dir == config.Direction.WEST:
            create_new_platoon = self.platoons[index[-1]].x[-1] < config.CANVAS_E - config.SAFE_DIS \
                                 or len(self.platoons[index[-1]].x) >= config.MAX_PLATOON_SIZE
        else:
            create_new_platoon = self.platoons[index[-1]].x[-1] > config.SAFE_DIS \
                                 or len(self.platoons[index[-1]].x) >= config.MAX_PLATOON_SIZE

        init_x = init_xs[start_dir.value]
        init_y = init_ys[start_dir.value]
        if create_new_platoon:
            p_new = car.Platoon(self.count, np.array([init_x]), np.array([init_y]), np.array([init_v]),
                                np.array([init_a]), start_dir, end_dir)
            self.platoons.append(p_new)
            self.count += 1
            platoon_show = [self.canvas.create_rectangle(
                init_x, init_y, init_x + config.CAR_LEN, init_y + config.CAR_WIDTH)]
            self.platoons_show.append(platoon_show)
        else:
            self.platoons[index[-1]].add_one_car(np.array([init_x]), np.array([init_y]),
                                                 np.array([init_v]), np.array([init_a]))
            self.platoons_show[index[-1]].append(
                self.canvas.create_rectangle(init_x, init_y, init_x + config.CAR_LEN, init_y + config.CAR_WIDTH))

    def create(self):
        lambda_ = 0.8
        if random.random() < lambda_*config.DT*np.exp(-lambda_*config.DT):
            start_dir = random.randint(0, 3)
            end_dir = random.randint(0, 3)
            while abs(end_dir - start_dir) == 2:
                end_dir = random.randint(0, 3)
            self.create_one_car(config.Direction(start_dir), config.Direction(end_dir))

    def cars_in_dir(self, start_dir, end_dir):                         # 沿着某个方向dir的车辆队列，根据前后距离排序
        index = []
        for i in range(len(self.platoons)):
            if self.platoons[i].start_dir == start_dir and self.platoons[i].end_dir == end_dir:
                index.append(i)

        for i in range(len(index) - 1):
            for j in range(len(index) - 1 - i):
                if compare(start_dir, self.platoons[index[j + 1]], self.platoons[index[j]]):
                    index[j], index[j + 1] = index[j + 1], index[j]

        return index

    def time_to_stop_line(self, phase, is_leader):
        dir = signal.phase_to_dir(phase)
        for_return = []
        for i in range(len(self.platoons)):
            if (self.platoons[i].start_dir == dir[0][0] and self.platoons[i].end_dir == dir[0][
                1]) or (self.platoons[i].start_dir == dir[1][0] and self.platoons[i].end_dir ==
                        dir[1][1]):
                leader_dis, tail_dis = self.platoons[i].dis_to_crossing()
                if is_leader:
                    for_return.append(leader_dis/10)
                else:
                    for_return.append(tail_dis/10)
        for_return.sort()
        return for_return

    def get_front_platoon(self, platoon_id, start_dir, end_dir):
        index = self.cars_in_dir(start_dir, end_dir)

        if self.platoons[index[0]].id == platoon_id:
            return -1

        for i in range(len(index) - 1):
            if self.platoons[index[i+1]].id == platoon_id:
                return self.platoons[index[i]]

    def find_platoon(self, p_id):
        result = -1
        for i in range(len(self.platoons)):
            if self.platoons[i].id == p_id:
                result = i
                break
        return result

    def step(self):
        done = (len(self.platoons) > 0)  # 表示整个场景的仿真是否完成

        current_phase = self.signal.get_current_phase()
        next_phase_to_line_time = self.time_to_stop_line((current_phase + 1) % 4, False)
        next_next_phase_to_line_time = self.time_to_stop_line((current_phase + 2) % 4, True)
        self.signal.cal_new_time(next_phase_to_line_time, next_next_phase_to_line_time)
        fill_color = ['green', 'yellow', 'red']
        for i in range(4):
            for j in range(3):
                start_dir = config.Direction(i)
                lane = j
                if j == 2:
                    lane = 3
                end_dir = config.Direction((i+lane) % 4)
                self.canvas.itemconfigure(self.signal_show[i][j], fill=fill_color[self.signal.get_color(start_dir, end_dir)])

        for i in range(len(self.platoons)):
            self.platoons[i].add_time()
            self.passed_crossing += self.platoons[i].cal_whether_pass()

        i = 0
        while i < len(self.platoons):
            time_left = self.get_signal_time_left()
            color = self.signal.get_color(self.platoons[i].start_dir, self.platoons[i].end_dir)
            about_to_green_ = self.signal.about_to_green(self.platoons[i].start_dir, self.platoons[i].end_dir)
            follow_signal_result = self.platoons[i].follow_signal(color, time_left, about_to_green_)
            front_platoon = self.get_front_platoon(self.platoons[i].id, self.platoons[i].start_dir, self.platoons[i].end_dir)
            self.platoons[i].follow_front_platoon(front_platoon, follow_signal_result)
            # update
            self.platoons[i].update()
            if self.platoons[i].reach_des():  # 只要有一个free platoon尚未到达目的地,就认为整个场景的仿真没有完成
                for j in range(self.platoons[i].get_num()):
                    self.canvas.delete(self.platoons_show[i][j])
                    # start_dir = self.platoons[i].start_dir
                    # end_dir = self.platoons[i].end_dir
                    # if end_dir.value - start_dir.value != 1 and end_dir.value - start_dir.value != -3:
                    #     print(self.platoons[i].time[j])
                del self.platoons[i]
                del self.platoons_show[i]
            else:
                done = False
                i = i + 1

        # 界面上的“方形”相应移动
        for i in range(len(self.platoons)):
            for j in range(self.platoons[i].get_num()):
                coord_s = self.canvas.coords(self.platoons_show[i][j])
                dx_int = round(self.platoons[i].x[j] - coord_s[0])
                dy_int = round(self.platoons[i].y[j] - coord_s[1])
                self.canvas.move(self.platoons_show[i][j], dx_int, dy_int)

        return done

    def render(self):
        time.sleep(0.01)
        self.update()

    def cal_queue_length(self):
        queue = 0
        for i in range(len(self.platoons)):
            start_dir = self.platoons[i].start_dir
            end_dir = self.platoons[i].end_dir
            if end_dir.value - start_dir.value == 1 or end_dir.value - start_dir.value == -3:
                continue
            color = self.signal.get_color(start_dir, end_dir)
            leader_dis, tail_dis = self.platoons[i].dis_to_crossing()
            if tail_dis > 0 and leader_dis < 100 and color != 0:
                queue += len(self.platoons[i].x)
        return queue


def compare(direction, plt1, plt2):                 # 按先后顺序排序，若plt1在plt2的前面，则返回True
    if direction == config.Direction.EAST:
        return plt1.x[0] > plt2.x[0]
    elif direction == config.Direction.WEST:
        return plt1.x[0] < plt2.x[0]
    elif direction == config.Direction.SOUTH:
        return plt1.y[0] > plt2.y[0]
    else:
        return plt1.y[0] < plt2.y[0]
