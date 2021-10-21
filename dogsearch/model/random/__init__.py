import time
from random import random, randint

from dogsearch.model import Model


class RandomModel(Model):
    def process(self, data, ext):
        res = {}
        res["is_animal_there"] = 1 if random() <= 0.4 else 0
        if res["is_animal_there"] == 1:
            res["is_it_a_dog"] = 1 if random() <= 0.3 else 0
            if res["is_it_a_dog"] == 1: # animal, which is a dog
                res["is_the_owner_there"] = 1 if random() <= 0.2 else 0
                res["color"] = randint(0, 2)
                res["tail"] = randint(0, 2)
            else: # animal, but not a dog
                res["is_the_owner_there"] = 2 # undefined
                res["color"] = 3 # undefined
                res["tail"] = 3 # undefined
        else:
            res["is_it_a_dog"] = 2 # undefined
            res["is_the_owner_there"] = 2 # undefined
            res["color"] = 3 # undefined
            res["tail"] = 3 # undefined
        time.sleep(randint(1, 10))
        return res
