""" Air Hockey Table State """
import json
import time

from .State import State


class AirHockeyTableState(State):
    """ Captures the state of the air hockey table """

    bot_state_name = "machine-state-bot"
    puck_state_name = "machine-state-puck"

    def __init__(self):
        # Define Initiate Constants
        self.__channel = "state-changed"
        self.__message = "state-changed"
        self.__sleep_time = 0.015

        self.__default_bot_position = {"x": 100, "y": 100}
        self.__default_puck_position = {"x": 100, "y": 100}

        # Subscribe to channel
        self.pubsub.subscribe(self.__channel)

    def publish(self, data, name="default"):
        """" Publish data to Redis """

        # Naming shortcuts
        if name == "bot":
            self.name = self.bot_state_name
        elif name == "puck":
            self.name = self.puck_state_name
        else:
            self.name = name

        # Put data in redis
        self.redis_serialize_set(self.name, data)
        self.redis.publish(self.__channel, self.__message)

        return None

    def subscribe(self, handle):
        """ Subscribe to channel in Redis """

        message = self.pubsub.get_message()

        if isinstance(message, dict) and (message["channel"].decode("utf-8") == self.__channel):

            # Handles issues when the positions are not loaded into redis, sets to defaults
            try:
                puck_state = json.loads(self.redis.get(self.puck_state_name))
            except TypeError:
                puck_state = self.__default_puck_position

            try:
                bot_state = json.loads(self.redis.get(self.bot_state_name))
            except TypeError:
                bot_state = self.__default_bot_position

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

    @property
    def default_bot_position(self):
        return self.__default_bot_position

    @default_bot_position.setter
    def default_bot_position(self, data):
        self.__default_bot_position = data
    
    @property
    def default_puck_position(self):
        return self.__default_puck_position

    @default_puck_position.setter
    def default_puck_position(self, data):
        self.__default_puck_position = data