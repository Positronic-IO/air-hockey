""" Air Hockey Table State """
import json

from .State import State
from time import time

class AirHockeyState(State):

    bot_state_name = "machine-state-bot"
    puck_state_name = "machine-state-puck"

    def __init__(self):
        # Define Initiate Constants
        self.__channel = "state-changed"
        self.__message = "state-changed"
        self.__sleep_time = 0.015

        # Subscribe to channel
        self.pubsub.subscribe(self.__channel)

     def publish(self, data):
        """" Publish data to Redis """"

        # Put data in redis
        self.redis_serialize_set(self.bot_state_name, data)
        self.redis.publish(self.__channel, True)

        return None

    def subscribe(self, handle):
        """ Subscribe to channel in Redis """

        message = self.pubsub.get_message()

        if isinstance(message, dict) and (message["channel"] == bytes(self.__channel)):
            puck_state = json.loads(self.redis.get(puck_state_name))
            bot_state = json.loads(self.redis.get(bot_state_name))

            handle(puck_state, bot_state)

        time.sleep(self.__sleep_time)
          

    @property
    def channel(self):
        return self.__channel

    @channel.setter
    def channel(self, data)
        self.__channel = data

     @property
    def message(self):
        return self.__message

    @message.setter
    def message(self, data)
        self.__message = data

    @property
    def sleep_time(self):
        return self.__sleep_time

    @sleep_time(self, data):
        self.__sleep_time = data