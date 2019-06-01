import config

'''
Phase setting
north-south straight and right: 0
north-south left: 1
east-west straight and right: 2
east-west left: 3
'''


class Signal:
    def __init__(self):
        self.time_left = 20
        self.current_phase = 0
        self.green_or_yellow = 0                                            # 0 for green, 1 for yellow

    def cal_new_time(self, next_queue_time, next_next_queue_time):
        self.time_left -= config.DT
        if self.green_or_yellow == 0:
            if self.time_left < config.DT/2:
                self.green_or_yellow = 1
                self.time_left = 3
        if self.green_or_yellow == 1:
            if self.time_left < config.DT/2:
                self.green_or_yellow = 0
                self.current_phase = (self.current_phase + 1) % 4
                self.time_left = cal_green_time(next_queue_time, next_next_queue_time)

    def get_color(self, start_dir, end_dir):
        phase = dir_to_phase(start_dir, end_dir)
        if phase == self.current_phase:
            return self.green_or_yellow
        else:
            return 2

    def get_current_phase(self):
        return self.current_phase

    def about_to_green(self, start_dir, end_dir):
        pre_phase = (dir_to_phase(start_dir, end_dir) - 1) % 4
        if pre_phase == self.current_phase and self.green_or_yellow == 0:
            return True
        else:
            return False

    def get_time_left(self):
        return self.time_left


def dir_to_phase(start_dir, end_dir):
    if end_dir.value - start_dir.value == -1 or end_dir.value - start_dir.value == 3:       # turn left
        if start_dir == config.Direction.NORTH or start_dir == config.Direction.SOUTH:
            return 1
        else:
            return 3
    else:                                                           # straight or turn right
        if start_dir == config.Direction.NORTH or start_dir == config.Direction.SOUTH:
            return 0
        else:
            return 2


def phase_to_dir(phase):                        # doesn't consider vehicles turning right
    if phase == 0:
        return [[config.Direction.NORTH, config.Direction.NORTH], [config.Direction.SOUTH, config.Direction.SOUTH]]
    elif phase == 1:
        return [[config.Direction.NORTH, config.Direction.WEST], [config.Direction.SOUTH, config.Direction.EAST]]
    elif phase == 2:
        return [[config.Direction.WEST, config.Direction.WEST], [config.Direction.EAST, config.Direction.EAST]]
    else:
        return [[config.Direction.WEST, config.Direction.SOUTH], [config.Direction.EAST, config.Direction.NORTH]]


def cal_green_time(next_phase_to_line_time, next_next_phase_to_line_time):
    if config.FIXED_TIMING:
        return 25
    else:
        np = len(next_phase_to_line_time)
        nq = len(next_next_phase_to_line_time)
        time_in_order = [0]*(np + nq)
        cost = [0]*(np + nq)
        p = q = i = 0
        while p < np and q < nq:
            while p < np and next_phase_to_line_time[p] < next_next_phase_to_line_time[q]:
                time_in_order[i] = next_phase_to_line_time[p]
                cost[i] = q + np - p - 1
                i += 1
                p += 1

            while q < nq and p < np and next_next_phase_to_line_time[q] < next_phase_to_line_time[p]:
                time_in_order[i] = next_next_phase_to_line_time[q]
                cost[i] = q + np - p - 1
                i += 1
                q += 1
        while p < np:
            time_in_order[i] = next_phase_to_line_time[p]
            cost[i] = nq + np - p - 1
            i += 1
            p += 1
        while q < nq:
            time_in_order[i] = next_next_phase_to_line_time[q]
            cost[i] = q
            i += 1
            q += 1

        min_cost = min(cost)
        arg_min_time = []
        for i in range(np + nq):
            if cost[i] == min_cost:
                arg_min_time.append(time_in_order[i])

        for_return = 0
        if max(arg_min_time) < config.MIN_GREEN_TIME:
            for_return = config.MIN_GREEN_TIME
        elif min(arg_min_time) > config.MAX_GREEN_TIME:
            for_return = config.MAX_GREEN_TIME
        else:
            for i in range(len(arg_min_time)):
                if config.MIN_GREEN_TIME < arg_min_time[i] < config.MAX_GREEN_TIME:
                    for_return = int(arg_min_time[i]) + 1
                    break
        return for_return
