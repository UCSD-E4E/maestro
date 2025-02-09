import os 
from kubernetes import client, config, utils
import eventlet
import socketio
from os import path
import yaml
import pandas as pd
import random
import shutil
import copy
from label_studio_ml.api import init_app
from maestro_model import MaestroModel, sio


print(os.environ['TRAINER_IMAGE'], os.environ['SERVER_URL'])


# This should handle the RESTful api requests from label studio
# but still serve the socketio system for manging the backend cluster
model_app = init_app(model_class=MaestroModel)
app = socketio.WSGIApp(sio, model_app, static_files={
    '/': {'content_type': 'text/html', 'filename': 'index.html'}
})

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 3000)), app)