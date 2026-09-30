from abc import ABC, abstractmethod

class Scheduler(ABC):
    @abstractmethod
    def __init__(self, cfg):
        pass

    ## Spins up a model class
    @abstractmethod
    def init_model(self, job_cfg):
        pass

    ## sends data to save for training
    @abstractmethod
    def send_data(self, job_cfg, model, data):
        pass

    ## trains model with active data
    ## Return loss
    @abstractmethod
    def run_batch(self, job_cfg, model):
        pass

    ## Spins down a model class
    @abstractmethod
    def deconstruct_model(self, job_cfg):
        pass

    ## Ask MAB for new model parameters
    @abstractmethod
    def query_model_MAB(self, model_citeria, next_citeria):
        pass

    ## Adjust current running jobs
    @abstractmethod
    def update_models(self, job_cfg, model):
        pass