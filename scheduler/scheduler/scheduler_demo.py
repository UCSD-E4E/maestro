import os 
from kubernetes import client, config, utils
import eventlet
import socketio


## DEBUG   
path = os.getcwd() 
dir_list = os.listdir(path) 

print("online", flush=True)
print(dir_list, flush=True)
## END DEBUG

config.load_incluster_config()
v1 = client.ApiClient()

sio = socketio.Server()
app = socketio.WSGIApp(sio, static_files={
    '/': {'content_type': 'text/html', 'filename': 'index.html'}
})

## Baseline server stuff I got from examples

@sio.event
def connect(sid, environ):
    print('connect ', sid, flush=True)

@sio.event
def my_message(sid, data):
    print('message ', data, flush=True)

@sio.event
def disconnect(sid):
    print('disconnect ', sid, flush=True)


## Key Functions for Schuduler

# Given the command from some admin portal
# Start processing data
@sio.on('start_test')
def spin_up_job(sid):
    yaml_file = 'job.yaml'
    #TODO: Fix permissions to allow for running jobs
    #utils.create_from_yaml(v1,yaml_file,verbose=True)
    

# job is connected and ready to start working
# start the job's actions with data
@sio.on('job_ready')
def start_job(sid, data):
    print("JOB IS READY TO START", flush=True)
    sio.emit('start_job', {'example_input': [0, 1, 2, 3, 4]})
    

@sio.on('job_done')
def clean_up_job(sid, data):
    print("random number from job", data["example_output"], sid, flush=True)
    # job has returned some data
    # print data return and spin down the job

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 3000)), app)