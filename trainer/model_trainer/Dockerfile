# Use an official Python runtime as a parent image.
FROM quay.io/jupyter/pytorch-notebook:cuda12-python-3.11.8

# data mount is own by root
# jupyter build sets users to jovyan
# This is probably worth fixing in the future

USER root
RUN usermod -aG root jovyan
USER jovyan
RUN groups

WORKDIR /home/jovyan/app
RUN echo "hello"
ADD . /home/jovyan/app
RUN ls -la .

RUN pip install --upgrade pip timm python-socketio requests
# RUN poetry install

#CMD ["poetry", "run", "python", "src/model_trainer.py"]

#RUN chmod 770 startup.sh
#RUN chmod -R 770 /data
RUN ls -la /data
RUN ls -la /
RUN touch /data/has_permissions.txt
CMD ["python", "src/model_trainer.py"] 