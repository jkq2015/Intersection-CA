import numpy as np
import config

import car
from env_crossing import Crossing
from RL_brain import DDPG


def simulate():
    # fo = open("plot.txt", 'a')
    count = 0
    while count < 6000:
        env.render()
        env.create()
        if config.IS_RL:
            env.cal_agent_free()
            a = []
            for i in range(len(env.agents)):
                a.append([0, 0])
            # choose action
            for i in range(len(env.agents)):
                a[i] = ddpg.choose_action(env.agents[i].observation)
                # 限制acc
                v0 = env.agents[i].platoons[0].v[0]
                v1 = env.agents[i].platoons[1].v[0]
                if env.agents[i].is_p0_min == 1:
                    a[i][0] = car.constrain_acc(a[i][0], v0)
                    a[i][1] = car.constrain_acc(a[i][1], v1)
                else:
                    a[i][1] = car.constrain_acc(a[i][1], v0)
                    a[i][0] = car.constrain_acc(a[i][0], v1)

            done = env.step(a)
        else:
            env.cal_in_region_free()
            done = env.step_fifo()
        print(count)
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
    train = False                       # false代表调用已经训练好的模型，true代表训练并将模型保存在Data文件夹下
# 由于目前的环境已经不是训练的环境了，所以这里不要改成true

    x0 = np.array([80, 64, 50, 35, 20]) - 10
    y0 = np.ones(5) * (config.CANVAS_E / 2 + 1.5*config.LANE_WIDTH)
    x1 = np.ones(5) * (config.CANVAS_E / 2 + 1.5 * config.LANE_WIDTH)
    x2 = np.array([20, 35, 50, 62, 76]) + config.right_side + config.START_DIS
    y2 = np.ones(5) * (config.CANVAS_E / 2 - 1.5*config.LANE_WIDTH)

    y1 = np.array([30, 45, 59, 74, 89]) + config.right_side + config.START_DIS
    y4 = np.array([89, 105, 118]) + config.right_side + config.START_DIS
    p0 = car.Platoon(0, x0, y0, 10*np.ones(5), np.zeros(5), config.Direction.RIGHT, config.Direction.RIGHT)
    p1 = car.Platoon(1, x1, y1, 10*np.ones(5), np.zeros(5), config.Direction.UP, config.Direction.UP)
    p2 = car.Platoon(2, x2, y2, 10*np.ones(5), np.zeros(5), config.Direction.LEFT, config.Direction.DOWN)

    p3 = car.Platoon(3, y2, x0, 10*np.ones(5), np.zeros(5), config.Direction.DOWN, config.Direction.RIGHT)
    p4 = car.Platoon(4, x1[0:3], y4, 10*np.ones(3), np.zeros(3), config.Direction.UP, config.Direction.UP)

    p01 = car.Platoon(11, x0[0:2], y0[0:2], 10*np.ones(2), np.zeros(2), config.Direction.RIGHT, config.Direction.RIGHT)
    p02 = car.Platoon(12, x0[2:5], y0[2:5], 10*np.ones(3), np.zeros(3), config.Direction.RIGHT, config.Direction.RIGHT)
    p11 = car.Platoon(13, x1[0:2], y1[0:2], 10*np.ones(2), np.zeros(2), config.Direction.UP, config.Direction.UP)
    p12 = car.Platoon(14, x1[3:5], y1[3:5], 10*np.ones(2), np.zeros(2), config.Direction.UP, config.Direction.UP)
    env = Crossing([])
    ddpg = DDPG(config.A_DIM, config.S_DIM, [config.A_MIN, config.A_MAX], train)

    env.after(100, simulate)

    env.mainloop()
