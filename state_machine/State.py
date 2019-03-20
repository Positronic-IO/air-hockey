""" General State Class """
import json

from abc import ABC, abstractclassmethod
from redis import StrictRedis
from redis.exceptions import DataError


class State(ABC):

    redis = StrictRedis()
    pubsub = redis.pubsub(ignore_subscribe_messages=True)

    def __init__(self):
        pass

    @abstractclassmethod
    def subscribe(self, **kwargs):
        pass

    @abstractclassmethod
    def publish(self, **kwargs):
        pass

    def redis_serialize_set(self, name, data):
        """ Put data in redis, serialize if need be """

        try:
            self.redis.set(name, data)
        except DataError:
            self.redis.set(name, json.dumps(data))
