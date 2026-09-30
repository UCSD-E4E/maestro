# Use an official Python runtime as a parent image.
FROM quay.io/jupyter/pytorch-notebook:cuda12-python-3.11.8

# data mount is own by root
# jupyter build sets users to jovyan
# This is probably worth fixing in the future
USER root
RUN pip install --upgrade pip timm python-socketio requests

WORKDIR /app
ADD . /app
RUN ls -la .

ENTRYPOINT ["python", "model_trainer/src/model_trainer.py"] 
# CMD ["python", "src/model_trainer.py"] 
# 