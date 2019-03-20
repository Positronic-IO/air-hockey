""" General State Class """
import json

from abc import ABC, abstractclassmethod
from redis import StrictRedis

class State(ABC):

    redis = StrictRedis()
    pubsub = self.redis.pubsub(ignore_subscribe_messages=True)

    def __init__(self):
        pass

    @abstractclassmethod
    def subscribe(self, **kwargs):
        pass

    @abstractclassmethod
    def publish(self, *8kwargs)
        pass
    
    @staticmethod
    def redis_serialize_set(name, data):
        """ Put data in redis, serialize if need be """

        try:
            self.redis.set(name, data)
        except DataError:
            self.redis.set(name, json.dumps(data))

    # @staticmethod
    # def redis_serialize_publish(name, )    