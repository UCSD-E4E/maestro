# Use an official Python runtime as a parent image.
FROM python

WORKDIR /app
ADD . /app
RUN ls /app

RUN pip install --upgrade pip poetry
RUN poetry install

CMD ["poetry", "run", "python", "src/model_trainer.py"]