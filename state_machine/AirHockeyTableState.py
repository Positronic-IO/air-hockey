""" Air Hockey Table State """
import json
import time

from .State import State


class AirHockeyTableState(State):

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
        """" Publish data to Redis """

        # Put data in redis
        self.redis_serialize_set(self.bot_state_name, data)
        self.redis.publish(self.__channel, self.__message)

        return None

    def subscribe(self, handle):
        """ Subscribe to channel in Redis """

        message = self.pubsub.get_message()
        
        if isinstance(message, dict) and (message["channel"].decode("utf-8") == self.__channel):
            puck_state = json.loads(self.redis.get(self.puck_state_name))
            bot_state = json.loads(self.redis.get(self.bot_state_name))

            print(puck_state, bot_state)
            handle(puck_state, bot_state)

        time.sleep(self.__sleep_time)

    @property
    def channel(self):
        return self.__channel

    @channel.setter
    def channel(self, data):
        self.__channel = data

    @property
    def message(self):
        return self.__message

    @message.setter
    def message(self, data):
        self.__message = data

    @property
    def sleep_time(self):
        return self.__sleep_time

    @sleep_time.setter
    def sleep_time(self, data):
        self.__sleep_time = data
