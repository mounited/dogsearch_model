from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def process(self, data, ext):
        pass

    @staticmethod
    def create(type):
        if type == "random":
            import dogsearch.model.random
            return dogsearch.model.random.RandomModel()
        elif type == "real":
            import dogsearch.model.real
            return dogsearch.model.real.RealModel()
        return None
