# Use an official Python runtime as a parent image.
FROM quay.io/jupyter/pytorch-notebook:cuda12-python-3.11.8

WORKDIR /app
ADD . /app
RUN ls -la /app

RUN pip install --upgrade pip timm python-socketio requests
# RUN poetry install

#CMD ["poetry", "run", "python", "src/model_trainer.py"]

#RUN chmod 777 startup.sh
CMD ["python", "src/model_trainer.py"] 
#"./startup.sh"