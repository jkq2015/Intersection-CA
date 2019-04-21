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
        color = self.get_signal_color()
        self.signal_show.append(self.canvas.create_line(
            config.CANVAS_E / 2, config.right_side, config.right_side, config.right_side, fill=fill_color[color[0]]))
        self.signal_show.append(self.canvas.create_line(
            config.left_side, config.left_side, config.CANVAS_E/2, config.left_side, fill=fill_color[color[1]]))
        self.signal_show.append(self.canvas.create_line(
            config.right_side, config.left_side, config.right_side, config.CANVAS_E/2, fill=fill_color[color[2]]))
        self.signal_show.append(self.canvas.create_line(
            config.left_side, config.CANVAS_E/2, config.left_side, config.right_side, fill=fill_color[color[3]]))

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

    def get_signal_color(self):
        return self.signal.get_color()

    def get_signal_time_left(self):
        return self.signal.get_time_left()

    def create_one_car(self, direction):
        index = self.cars_in_dir(direction)
        init_xs = [config.CANVAS_E / 2 + 1.5*config.LANE_WIDTH, config.CANVAS_E / 2 - 1.5 * config.LANE_WIDTH,
                   config.CANVAS_E - 1, 1]
        init_ys = [config.CANVAS_E - 1, 1, config.CANVAS_E / 2 - 1.5 * config.LANE_WIDTH,
                   config.CANVAS_E / 2 + 1.5*config.LANE_WIDTH]
        init_a = 0.
        init_v = config.V_MAX

        if len(index) == 0:
            create_new_platoon = True
        elif direction == config.Direction.NORTH:
            create_new_platoon = (self.platoons[index[-1]].y[-1] < config.CANVAS_E - config.SAFE_DIS) \
                                 or len(self.platoons[index[-1]].x) >= config.MAX_PLATOON_SIZE

        elif direction == config.Direction.SOUTH:
            create_new_platoon = self.platoons[index[-1]].y[-1] > config.SAFE_DIS \
                                 or len(self.platoons[index[-1]].x) >= config.MAX_PLATOON_SIZE
        elif direction == config.Direction.WEST:
            create_new_platoon = self.platoons[index[-1]].x[-1] < config.CANVAS_E - config.SAFE_DIS \
                                 or len(self.platoons[index[-1]].x) >= config.MAX_PLATOON_SIZE
        else:
            create_new_platoon = self.platoons[index[-1]].x[-1] > config.SAFE_DIS \
                                 or len(self.platoons[index[-1]].x) >= config.MAX_PLATOON_SIZE

        init_x = init_xs[direction.value]
        init_y = init_ys[direction.value]
        if create_new_platoon:
            p_new = car.Platoon(self.count, np.array([init_x]), np.array([init_y]), np.array([init_v]),
                                np.array([init_a]), direction, direction)
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
        lambda_ = 0.5
        if random.random() < lambda_*config.DT*np.exp(-lambda_*config.DT):
            r = random.randint(0, 3)
            self.create_one_car(config.Direction(r))

    def cars_in_dir(self, direction):                         # 沿着某个方向dir的车辆队列，根据前后距离排序
        index = []
        for i in range(len(self.platoons)):
            if self.platoons[i].start_dir == direction:
                index.append(i)

        for i in range(len(index) - 1):
            for j in range(len(index) - 1 - i):
                if compare(direction, self.platoons[index[j + 1]], self.platoons[index[j]]):
                    index[j], index[j + 1] = index[j + 1], index[j]

        return index

    def get_front_platoon(self, platoon_id, direction):
        index = self.cars_in_dir(direction)

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

        old_color = self.get_signal_color()
        self.signal.cal_new_time()
        new_color = self.get_signal_color()
        fill_color = ['green', 'yellow', 'red']
        for i in range(len(old_color)):
            if new_color[i] != old_color[i]:
                self.canvas.itemconfigure(self.signal_show[i], fill=fill_color[new_color[i]])

        for i in range(len(self.platoons)):
            self.platoons[i].add_time()
            self.passed_crossing += self.platoons[i].cal_whether_pass()

        i = 0
        while i < len(self.platoons):
            time_left = self.signal.get_time_left()
            index = self.platoons[i].start_dir.value
            follow_signal_result = self.platoons[i].follow_signal(new_color[index], time_left[index])
            front_platoon = self.get_front_platoon(self.platoons[i].id, self.platoons[i].start_dir)
            self.platoons[i].follow_front_platoon(front_platoon, follow_signal_result)
            # update
            self.platoons[i].update()
            if self.platoons[i].reach_des():  # 只要有一个free platoon尚未到达目的地,就认为整个场景的仿真没有完成
                for j in range(self.platoons[i].get_num()):
                    self.canvas.delete(self.platoons_show[i][j])
                    # fo.write(str(self.platoons[i].time[j]) + '\n')
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
        time.sleep(0.02)
        self.update()


def compare(direction, plt1, plt2):                 # 按先后顺序排序，若plt1在plt2的前面，则返回True
    if direction == config.Direction.EAST:
        return plt1.x[0] > plt2.x[0]
    elif direction == config.Direction.WEST:
        return plt1.x[0] < plt2.x[0]
    elif direction == config.Direction.SOUTH:
        return plt1.y[0] > plt2.y[0]
    else:
        return plt1.y[0] < plt2.y[0]
