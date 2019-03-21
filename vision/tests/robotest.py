import robot
import time

fluffy = robot.Robot()

ct=0
dist=10
while True:
    if ct == 10:
        break
    fluffy.goto(0, dist)
    dist += 10
    ct += 1
    time.sleep(0.15)

print("Done")