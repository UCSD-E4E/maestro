import os 
from kubernetes import client, config, utils
import eventlet
import socketio
from os import path
import yaml
import pandas as pd
import time

# while True:
#     time.sleep(1)


## DEBUG   
path = os.getcwd() 
dir_list = os.listdir(path) 

print("online", flush=True)
print(dir_list, flush=True)
## END DEBUG

config.load_incluster_config()
v1 = client.ApiClient()
batch_v1_api = client.BatchV1Api()
yaml_file = 'scheduler/job.yaml'

sio = socketio.Server()
app = socketio.WSGIApp(sio, static_files={
    '/': {'content_type': 'text/html', 'filename': 'index.html'}
})

cfg = {
    "data_path": "/data/",
    "model_name": "efficientnet_b0",
    "model_checkpoint": "/data/model.pt",
    "learning_rate": 0.001
}

active_jobs = {}
loss = []

# Set Up The Dataset for Training
df = pd.read_csv("/data/classification.csv")
df = df.sample(frac = 1)
df["file"] = df["ID"] + ".jpg"
# This is going to be used to keep track what has already been shown.
# In the future this will need to be far more complicated...
active_idx = 0

def get_next_label():
    row = df.sample(1)
    return {
        "file_path": row["file"],
        "label": row["labels"]
    }

def update_labeled_points(idx):
    pass
# Writing this, we probably need a custom dataset class to manage this kind of thing...



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
    yaml_file = 'scheduler/job.yaml'
    #TODO: Fix permissions to allow for running jobs
    with open(yaml_file) as f:
        job_dict = yaml.safe_load(f)
        print(job_dict)
        job_dict["metadata"]["name"] = f"model-training-{len(active_jobs)}"
        job = utils.create_from_dict(v1, job_dict,verbose=True, namespace="krg-maestro")[0]
        print(job)
        print(dir(job))
        print(job.metadata.name)
    

# job is connected and ready to start working
# start the job's actions with data
@sio.on('job_ready')
def start_job(sid, data):
    print("JOB READY", data, flush=True)
    active_jobs[data["pod-name"]] = "running"
    print("JOB IS READY TO START", flush=True)

    batch = get_next_label()
    data = {
        "batch": batch,
        "cfg": cfg
    }
    sio.emit('start_job', data)
    

@sio.on('job_done')
def clean_up_job(sid, data):
    print("got loss", data["loss"], data["pod-name"], active_jobs, sid, flush=True)
    del active_jobs[data["pod-name"]]
    loss.append(float(data["loss"]))
    # job has returned some data
    # print data return and spin down the job
    batch_v1_api.delete_namespaced_job(data["PODNAME"], "krg-maestro")
    spin_up_job(None)

@sio.on('get_loss')
def get_loss(sid):
    sio.emit('sending_loss', {"losses": loss})

if __name__ == '__main__':

    spin_up_job(None) 
    # # RUN JOB EXAMPLE!
    # with open(yaml_file) as f:
    #     job_dict = yaml.safe_load(f)
    #     print(job_dict)
    #     job_dict["metadata"]["name"] = "model-training-1"
    #     job = utils.create_from_dict(v1, job_dict,verbose=True, namespace="krg-maestro")[0]
    #     print(job)
    #     print(dir(job))
    #     print(job.metadata.name)


    #print("run once")
    #utils.create_from_yaml(v1,yaml_file,verbose=True, namespace="krg-maestro")
    #print("success!")
    eventlet.wsgi.server(eventlet.listen(('', 3000)), app)