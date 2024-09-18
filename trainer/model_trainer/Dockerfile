# Use an official Python runtime as a parent image.
FROM python

WORKDIR /app
ADD . /app
RUN ls -la /app

RUN pip install --upgrade pip poetry
# RUN poetry install

#CMD ["poetry", "run", "python", "src/model_trainer.py"]

RUN chmod 777 startup.sh
CMD ["./startup.sh"]