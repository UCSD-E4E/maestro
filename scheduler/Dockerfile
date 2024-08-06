FROM python

WORKDIR /app
ADD . /app

RUN pip install --upgrade pip poetry
RUN poetry install

EXPOSE 3000

CMD ["poetry", "run", "python", "scheduler/scheduler_demo.py"]