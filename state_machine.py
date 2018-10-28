import numpy as np
import redis
import time
import json
import math
from vision import robot

#sleep_time = 0.015
sleep_time = 0.1
fast = 10
slow = 5

fluffy = robot.Robot()
fluffy.speed_low = chr(255)
fluffy.speed_high = chr(10)
fluffy.accel_low = chr(255)
fluffy.accel_high = chr(10)

r = redis.StrictRedis(host="localhost", port=6379, db=0)
p = r.pubsub(ignore_subscribe_messages=True)

min_y = 450

def meet_the_puck(puck_state, bot_state):
    speed = fast
    puck = json.loads(puck_state)
    bot = json.loads(bot_state)

    #find horiz and vertical distances between puck and bot
    dx = float(bot['x']) - float(puck['x'])
    dy = float(bot['y']) - float(puck['y'])
    
    # todo: 
    #   if puck is behind bot, do not go towards puck, go around.
        
    
    #get the hypotenuse
    d = math.hypot(dx, dy)
    #print("d=",d)
    #if d > 100:
    #    speed = slow
    
    if abs(d) > 30:
        # calculate the change to the position
        cx = min(speed * dx/d, dx)
        cy = min(speed * dy/d, dy)
        
        new_b = bot
        new_b['x'] = int(puck['x'])
        new_b['y'] = int(puck['y'])
        
        #if new_b['y'] < min_y:
        #    new_b['y'] = int(min_y)

        print(new_b, d)
        fluffy.goto(new_b['x'], new_b['y'])

        #r.set("machine-state-bot", json.dumps(new_b))
        #r.publish('state-changed', True)

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