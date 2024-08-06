from random import choices

from scheduler import Scheduler

from kubernetes import client, config, utils
import eventlet
import socketio

class KubeScheduler(Scheduler):
    #TODO: Should this be programmatically defined instead?
    yaml_file = 'job.yaml' #Job Template for Running Models
    sio = socketio.Server()

    def __init__(self, cfg, model_list):
        config.load_incluster_config()
        self.admin_sid = None
        self.v1 = client.ApiClient()
        self.app = socketio.WSGIApp(self.sio, static_files={
            '/': {'content_type': 'text/html', 'filename': 'index.html'}
        })

        self.possible_models = model_list
        self.model_jobs = []

        self.data_send_queue = []
        self.data_request_queue

        # Start server and init the frist two jobs
        self.start_server()

        self.current_model, self.next_model = choices(model_list, k = 2)
        self.init_model(self.current_model)
        self.init_model(self.next_model)
    
    ## Begin Operations
    async def start_server(self):
        eventlet.wsgi.server(eventlet.listen(('', 3000)), self.app)

    # Given the command from some admin portal
    # Start processing data

    # TODO: Some questions about archiecture here
    # Does it make sense to have the queue send data
    # Or have the schuduler request data from the queue
    # I'm thinking the queue sends data, schuduler maintains an internal queue?
    # Notes: "Label Queue" makes more operational sense as a labeling / schuduler middle man
    # Schuduler should have its own queue in addition to label queue
    # Label queue exists to handle parrell users. 
    @sio.on('data_ready_to_train')
    def queue_next_data(self, sid, data):
        if self.admin_sid == None:
            self.admin_sid = sid
        elif self.admin_sid != sid:
            # One queue per schuduler?
            return 403 

        if 
        
        self.data_send_queue.push(data)
        
    ## Spins up a model class
    def init_model(self, job_cfg):
        #TODO: prep config for new model job

        # Start the model job
        #TODO: Fix permissions to allow for running jobs
        #utils.create_from_yaml(v1,yaml_file,verbose=True)
        pass

    ## Models should request data
    def handle_data_request(self, job_cfg, model, data):
        pass

    ## trains model with active data
    ## Return loss
    def run_batch(self, job_cfg, model):
        pass

    ## Spins down a model class
    def deconstruct_model(self, job_cfg):
        pass

    ## Ask MAB for new model parameters
    def query_model_MAB(self, model_citeria, next_citeria):
        pass

    ## Adjust current running jobs
    def update_models(self, job_cfg, model):
        pass