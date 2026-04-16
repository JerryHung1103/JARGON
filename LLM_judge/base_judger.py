from abc import ABC, abstractmethod
class BaseJudger(ABC):

    def __init__(self, name: str):
        self.name = name
        self.rationale = None

    @abstractmethod
    def is_harmful(self, query: str, response: str) -> int:
        pass

    def get_rationale(self):
        return self.rationale 

    def get_name(self):
        return self.name 
