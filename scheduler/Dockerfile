FROM quay.io/jupyter/pytorch-notebook:cuda12-python-3.11.8

WORKDIR /app
ADD . /app

RUN pip install --upgrade pip poetry
RUN poetry install

EXPOSE 3000

CMD ["poetry", "run", "python", "scheduler/label_studio_scheduler.py"]