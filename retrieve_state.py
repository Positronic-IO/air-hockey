""" Logic to update robot state """
import json
import logging

import numpy as np

from state_machine import AirHockeyTableState
from vision import robot

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

# Configure robot
fluffy = robot.Robot()
fluffy.speed_low = chr(255)
fluffy.speed_high = chr(10)
fluffy.accel_low = chr(255)
fluffy.accel_high = chr(10)

# Define Table Constants
TABLE_HEIGHT = 212  # (px)
TABLE_WIDTH = 411  # (px)
PUCK_RADIUS = 10
SAFETY_BORDER = 2  # (px)
MIN_X = 120 - 2 * PUCK_RADIUS
MIN_Y = (SAFETY_BORDER, TABLE_HEIGHT - SAFETY_BORDER)

# Initialize state
state = AirHockeyTableState()
state.sleep_time = 0.015

# Default Position
state.default_bot_position = {"x": 50, "y": int(TABLE_HEIGHT / 2)}
state.default_puck_position = {"x": TABLE_WIDTH - 50, "y": int(TABLE_HEIGHT / 2)}

# Bot speed
FAST = 100
SLOW = 50


def normalize(point):
    """ Normalize point """

    offset = json.loads(state.redis.get("table_offset"))
    norm_x = max(0, int(point["x"]) - int(offset["x"]))
    norm_y = max(0, int(point["y"]) - int(offset["y"]))
    new_point = {"x": norm_x, "y": norm_y}
    return new_point


def meet_the_puck(puck_state, bot_state):
    speed = FAST

    # Normalize state by offset
    # bot_state = normalize(bot_state)
    # puck_state = normalize(puck_state)

    # find horiz and vertical distances between puck and bot
    dx = float(bot_state['x']) - float(puck_state['x'])
    dy = float(bot_state['y']) - float(puck_state['y'])

    # todo:
    #   if puck is behind bot, do not go towards puck, go around.

    # get the hypotenuse
    d = np.hypot(dx, dy)

    if d > 100:
        speed = SLOW

    if abs(d) > 30:
        # calculate the change to the position
        cx = min(speed * dx/d, dx)
        cy = min(speed * dy/d, dy)

        # Calculage new state
        updated_bot_state = bot_state
        updated_bot_state['x'] -= int(cx)
        updated_bot_state['y'] -= int(cy)

        # Enforce boundary
        if updated_bot_state['x'] > MIN_X:
            updated_bot_state['x'] = int(MIN_X)

        # Safety border (Robot does not go all the way to edge of table)
        if updated_bot_state['y'] < MIN_Y[0]:
            updated_bot_state['y'] = MIN_Y[0]
        elif updated_bot_state['y'] > MIN_Y[1]:
            updated_bot_state['y'] = MIN_Y[1]

        # Send directions to the robot
        # fluffy.goto(updated_bot_state['x'], updated_bot_state['y'])

        # Update new position
        state.publish(name="bot", data=updated_bot_state)


if __name__ == "__main__":
    logger.info(f"Updating the state of {fluffy}")
    logger.info(f"Robot responsiveness: {state.sleep_time}")

    # If the table offset and dimensions have been calculated, display. Else, zero offset
    try:
        logger.info(
            f"Table offset: {json.loads(state.redis.get('table_offset'))}")

        logger.info(
            f"Table dimensions: {json.loads(state.redis.get('table_dimensions'))}")
    except TypeError:
        logger.info("Table offset: {'x': 0, 'y': 0}")
        logger.info(
            "Table dimensions are undefined.")

    # event loop for bot
    while True:
        state.subscribe(handle=meet_the_puck)
