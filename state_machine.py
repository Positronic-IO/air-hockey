import redis
import time

r = redis.StrictRedis(host='localhost', port=6379, db=0)
p = r.pubsub(ignore_subscribe_messages=True)

# 2 events, bot position and puck position.
p.subscribe('bot_pos', 'puck_pos')
p.get_message()


# event loop for state machine...
while True:
    message = p.get_message()
    if message:
        if message['channel'] == 'bot_pos':
            print 'New Bot Position: ' + message['data']
        elif message['channel'] == 'puck_pos':
            print 'New Puck Position: ' + message['data']
        else:
            print 'Message not recognized'
    time.sleep(0.001)