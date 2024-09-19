import random
import socketio
import os
from model import Trainer

print(os.environ)
pod_name = os.environ['pod-name']
#import time 
# while True: 
#     time.sleep(100)
#     print("loop")

print("RUNNING PYTHON!!!", flush=True)
sio = socketio.Client()

@sio.event
def connect():
    print('connection established', flush=True)
    sio.emit('job_ready', {'pod-name': pod_name})

@sio.on('start_job')
def trigger_test(data):
    print(data, flush=True)
    # Expect batch to be a list of directory of file and label
    batch = data["batch"]
    cfg = data["cfg"]
    trainer = Trainer(cfg, batch)
    trainer.build()
    loss = trainer.train()

    # number = random.randrange(0, 100)
    print("Job Done", loss, flush=True)
    sio.emit('job_done', {'loss': loss, "pod-name": pod_name})
    sio.disconnect()

@sio.event
def my_message(data):
    print('message received with ', data, flush=True)
    sio.emit('my response', {'response': 'my response'})

@sio.event
def disconnect():
    print('disconnected from server', flush=True)   

sio.connect('https://test-service2.nrp-nautilus.io')
sio.wait()



