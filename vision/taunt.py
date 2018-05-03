import robot
import time
import random

fluffy = robot.Robot()

fluffy.speed_low = chr(255)
fluffy.speed_high = chr(10)

fluffy.accel_low = chr(255)
fluffy.accel_high = chr(10)

x_range = [50, 150]

def set_left(x_range):
    x_range[0] = 55
    x_range[1] = 150

def set_right(x_range):
    x_range[0] = 350
    x_range[1] = 480

def toggle_side(x_range):
    print("Toggle")
    if toggle_side.left:
        set_right(x_range)
        toggle_side.left = False
    else:
        set_left(x_range)
        toggle_side.left = True
toggle_side.left = True

x = 265
y = 0

set_left(x_range)
left = True

ops = 0

while True:
    try:
        old_x = x
        old_y = y
        x = random.randint(x_range[0], x_range[1])
        y = random.randint(50, 300)
        print("GOTO: (" + str(x) + "," + str(y)+")")
        fluffy.goto(x, y)
        wait = (abs(old_x - x) + abs(old_y - y)) * .005 + .02
        print("Sleeping for: " + str(wait))
        time.sleep(wait)
        if ops > 10:
            toggle_side(x_range)
            ops = 0
    except KeyboardInterrupt:
        break
    ops += 1


time.sleep(2)
fluffy.goto(265, 100)
time.sleep(2)
fluffy.goto(265, 0)
print("Done")