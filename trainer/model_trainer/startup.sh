#!/bin/bash

# Note: Goal of this is to reuse a poetry enviroment with large size
# /data/ is the mount of the PVC
if  [ ! -d "/data/model_env" ]; then
    mkdir /data/model_env
fi

# Move default installation location in poetry to PVC and run
poetry config virtualenvs.path /data/model_env
poetry install --no-root
poetry run python src/model_trainer.py