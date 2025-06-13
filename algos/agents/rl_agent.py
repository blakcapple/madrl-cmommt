from abc import ABC, abstractmethod
class RLAgent(ABC):
    
    def __init__(self):
        pass
    
    @abstractmethod
    def select_actions(self):
        pass
    
    @abstractmethod
    def load_weight(self):
        pass
    