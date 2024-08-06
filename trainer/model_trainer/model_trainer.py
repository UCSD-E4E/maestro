import random
import socketio

print("RUNNING PYTHON!!!", flush=True)
sio = socketio.Client()

@sio.event
def connect():
    print('connection established', flush=True)
    sio.emit('job_ready', {'response': 'my response'})

@sio.on('start_job')
def trigger_test(data):
    print(data['example_input'], flush=True)
    number = random.randrange(0, 100)
    print("random number", number, flush=True)
    sio.emit('job_done', {'example_output': number})

@sio.event
def my_message(data):
    print('message received with ', data, flush=True)
    sio.emit('my response', {'response': 'my response'})

@sio.event
def disconnect():
    print('disconnected from server', flush=True)   

sio.connect('https://test-service2.nrp-nautilus.io')
sio.wait()



