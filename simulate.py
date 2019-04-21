import numpy as np
import config

import car
from env_crossing import Crossing


def simulate():
    # fo = open("plot.txt", 'a')
    count = 0
    while count < 6000:
        env.render()
        env.create()
        done = env.step()
        print(count, env.get_signal_color(), env.get_signal_time_left())
        count += 1
        # print(env.platoons[0].a[0])
        # if len(env.platoons) == 3:
        #     fo.write(str(env.platoons[0].a[0]) + '\t' + str(env.platoons[1].a[0]) + '\t' + str(env.platoons[2].a[0])
        #              + '\t' + str(env.platoons[0].v[0]) + '\t' + str(env.platoons[1].v[0])
        #              + '\t' + str(env.platoons[2].v[0]) + '\n')

        if done:
            # fo.close()
            break


if __name__ == "__main__":

    x0 = np.array([80, 64, 50, 35, 20]) - 10
    y0 = np.ones(5) * (config.CANVAS_E / 2 + 1.5*config.LANE_WIDTH)
    x1 = np.ones(5) * (config.CANVAS_E / 2 + 1.5 * config.LANE_WIDTH)
    x2 = np.array([20, 35, 50, 62, 76]) + config.right_side
    y2 = np.ones(5) * (config.CANVAS_E / 2 - 1.5*config.LANE_WIDTH)

    y1 = np.array([30, 45, 59, 74, 89]) + config.right_side
    y4 = np.array([89, 105, 118]) + config.right_side
    p0 = car.Platoon(0, x0, y0, 10 * np.ones(5), np.zeros(5), config.Direction.EAST, config.Direction.EAST)
    p1 = car.Platoon(1, x1, y1, 10 * np.ones(5), np.zeros(5), config.Direction.NORTH, config.Direction.NORTH)
    p2 = car.Platoon(2, x2, y2, 10 * np.ones(5), np.zeros(5), config.Direction.WEST, config.Direction.SOUTH)

    p3 = car.Platoon(3, y2, x0, 10 * np.ones(5), np.zeros(5), config.Direction.SOUTH, config.Direction.EAST)
    p4 = car.Platoon(4, x1[0:3], y4, 10 * np.ones(3), np.zeros(3), config.Direction.NORTH, config.Direction.NORTH)

    p01 = car.Platoon(11, x0[0:2], y0[0:2], 10 * np.ones(2), np.zeros(2), config.Direction.EAST, config.Direction.EAST)
    p02 = car.Platoon(12, x0[2:5], y0[2:5], 10 * np.ones(3), np.zeros(3), config.Direction.EAST, config.Direction.EAST)
    p11 = car.Platoon(13, x1[0:2], y1[0:2], 10 * np.ones(2), np.zeros(2), config.Direction.NORTH, config.Direction.NORTH)
    p12 = car.Platoon(14, x1[3:5], y1[3:5], 10 * np.ones(2), np.zeros(2), config.Direction.NORTH, config.Direction.NORTH)
    env = Crossing([])

    env.after(100, simulate)

    env.mainloop()
