import numpy as np
import redis
import time
import json
import math

r = redis.StrictRedis(host="localhost", port=6379, db=0)
p = r.pubsub(ignore_subscribe_messages=True)

def meet_the_puck(puck_state, bot_state):
    speed = 5
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
    
    if abs(d) >5:
        # calculate the change to the position
        cx = speed * dx/d
        cy = speed * dy/d
        
        new_b = b
        new_b['x'] -= cx
        new_b['y'] -= cy
        
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
        time.sleep(0.015)