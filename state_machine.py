import numpy as np
import redis
import time
import json
import math
from vision import robot
from state_machine import AirHockeyTable

#sleep_time = 0.015
sleep_time = 0.015
fast = 100
slow = 50

fluffy = robot.Robot()
fluffy.speed_low = chr(255)
fluffy.speed_high = chr(10)
fluffy.accel_low = chr(255)
fluffy.accel_high = chr(10)

state = AirHockeyTable()

min_y = 450


def meet_the_puck(puck_state, bot_state):
    speed = fast
    p = puck_state
    b = bot_state

    # find horiz and vertical distances between puck and bot
    dx = float(b['x']) - float(p['x'])
    dy = float(b['y']) - float(p['y'])

    # todo:
    #   if puck is behind bot, do not go towards puck, go around.

    # get the hypotenuse
    d = math.hypot(dx, dy)

    if d > 100:
        speed = slow

    if abs(d) > 30:
        # calculate the change to the position
        cx = min(speed * dx/d, dx)
        cy = min(speed * dy/d, dy)

        new_b = b
        new_b['x'] -= int(cx)
        new_b['y'] -= int(cy)

        if new_b['y'] < min_y:
            new_b['y'] = int(min_y)

        # fluffy.goto(new_b['x'], new_b['y'])

        state.publish(data=new_b)


if __name__ == "__main__":

    # event loop for bot
    while True:
        state.subscribe(handle=meet_the_puck)
