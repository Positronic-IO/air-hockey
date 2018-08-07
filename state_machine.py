import numpy as np
import redis
import time
import json
import math
from vision import robot

#sleep_time = 0.015
sleep_time = 0.015

fluffy = robot.Robot()
fluffy.speed_low = chr(255)
fluffy.speed_high = chr(10)
fluffy.accel_low = chr(255)
fluffy.accel_high = chr(10)

r = redis.StrictRedis(host="localhost", port=6379, db=0)
p = r.pubsub(ignore_subscribe_messages=True)

min_y = 450

def meet_the_puck(puck_state, bot_state):
    speed = 30
    p = json.loads(puck_state)
    b = json.loads(bot_state)

    #find horiz and vertical distances between puck and bot
    dx = float(b['x']) - float(p['x'])
    dy = float(b['y']) - float(p['y'])
    
    # todo: 
    #   if puck is behind bot, do not go towards puck, go around.
        
    
    #get the hypotenuse
    d = math.hypot(dx, dy)
    
    if d > 100:
        speed = 10
    
    if abs(d) > 30:
        # calculate the change to the position
        cx = speed * dx/d
        cy = speed * dy/d
        
        new_b = b
        new_b['x'] -= int(cx)
        new_b['y'] -= int(cy)
        
        if new_b['y'] < min_y:
            new_b['y'] = int(min_y)

        fluffy.goto(new_b['x'], new_b['y'])

        r.set("machine-state-bot", json.dumps(new_b))
        r.publish('state-changed', True)

p.subscribe('state-changed')

# event loop for bot
while True:
    message = p.get_message()
    if message:
        if message['channel'] == 'state-changed':
            puck_state = r.get('machine-state-puck')
            bot_state = r.get('machine-state-bot')
            meet_the_puck(puck_state, bot_state)
        time.sleep(sleep_time)