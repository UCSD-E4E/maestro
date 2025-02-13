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

## DEBUG
print(os.environ)


## DEMO FOR STARTING A JOB
config.load_incluster_config()
v1 = client.ApiClient()
batch_v1_api = client.BatchV1Api()
yaml_file = 'scheduler/job.yaml'

#https://github.com/kubernetes-client/python/issues/363


with open(yaml_file) as f:
    job_dict = yaml.safe_load(f)
    print(job_dict)

    uuid =  os.environ["HOSTNAME"].split("-maestro-scheduler")[0]
    job_name =  f"{uuid}-model-training-1"
    namespace = os.environ["POD_NAMESPACE"]
    job_image = os.environ["TRAINER_IMAGE"]
    url = os.environ['SERVER_URL']
    pvc_name = f"{uuid}-maestro-data-pvc"

    job_dict["metadata"]["name"] = job_name
    job_dict["metadata"]["labels"]["group"] = uuid
    job_dict["spec"]["template"]["metadata"]["labels"]["k8s-app"] = job_name
    job_dict["spec"]["template"]["metadata"]["name"] = job_name
    job_dict["spec"]["template"]["spec"]["containers"][0]["image"] = job_image
    job_dict["spec"]["template"]["spec"]["containers"][0]["env"][1]["value"] = url
    job_dict["spec"]["template"]["spec"]["containers"][0]["volumeMounts"][0][
        "name"
    ] = pvc_name
    job_dict["spec"]["template"]["spec"]["volumes"][0]["name"] = pvc_name
    job_dict["spec"]["template"]["spec"]["volumes"][0]["persistentVolumeClaim"][
        "claimName"
    ] = pvc_name
    job = utils.create_from_dict(v1, job_dict,verbose=True, namespace=namespace)[0]
    print("starting", job.metadata.name)


# This should handle the RESTful api requests from label studio
# but still serve the socketio system for manging the backend cluster
model_app = init_app(model_class=MaestroModel)
app = socketio.WSGIApp(sio, model_app, static_files={
    '/': {'content_type': 'text/html', 'filename': 'index.html'}
})

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 3000)), app)