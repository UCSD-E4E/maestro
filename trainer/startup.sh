# #!/bin/bash
# # Move default installation location in poetry to PVC and run
# poetry config virtualenvs.path /data/model_env

# # Note: Goal of this is to reuse a poetry enviroment with large size
# # /data/ is the mount of the PVC
# echo "check env location"
# if  [ ! -d "/data/model_env" ]; then
#     mkdir /data/model_env
    
#     echo "start install"
#     poetry install --no-root
#     echo "end install"
# fi
# echo "env setup done, running code"
#poetry run 

whoami
python src/model_trainer.py