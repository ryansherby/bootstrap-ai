import abc
from configs import CONFIG


class BaseModel(abc.ABC):
    
    
    def __init__(self,gpu_number,device='cuda'):
        
        
        self.gpu_number = gpu_number
        self.device = device
        
        
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        pass
    
    
    @classmethod
    def list_processes(cls):
        
        return [cls.name]