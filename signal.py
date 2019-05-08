import config


class Signal:
    def __init__(self):
        self.total_time = [20, 3, 69]
        self.color = [[0, 0, 2], [2, 2, 2], [0, 0, 2], [2, 2, 2]]           # 0 for green, 1 for yellow, 2 for red
        self.time_left = [[20, 20, 23], [46, 46, 69], [20, 20, 23], [46, 46, 69]]

    def cal_new_time(self):
        for i in range(4):
            for j in range(3):
                self.time_left[i][j] -= config.DT
                if self.time_left[i][j] < config.DT/2:
                    self.color[i][j] = (self.color[i][j] + 1) % 3
                    self.time_left[i][j] = self.total_time[self.color[i][j]]

    def get_color(self):
        for_return = []
        for i in range(4):
            for_return.append(self.color[i].copy())
        return for_return

    def get_time_left(self):
        return self.time_left
