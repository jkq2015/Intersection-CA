import config


class Signal:
    def __init__(self):
        self.total_time = [30, 5, 35]
        self.color = [0, 0, 2, 2]                 # 0 for green, 1 for yellow, 2 for red
        self.time_left = [self.total_time[self.color[i]] for i in range(4)]

    def cal_new_time(self):
        for i in range(4):
            self.time_left[i] -= config.DT
            if self.time_left[i] < config.DT/2:
                self.color[i] = (self.color[i] + 1) % 3
                self.time_left[i] = self.total_time[self.color[i]]

    def get_color(self):
        for_return = [self.color[i] for i in range(4)]
        return for_return

    def get_time_left(self):
        return self.time_left
