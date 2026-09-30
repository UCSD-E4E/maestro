import uuid
import json

DEFAULT_CFG = {
    "UUID": str(uuid.uuid4()),
    "scheduler_path": "",
    "trainer_path": "",
    "ingress_url": "",
    "namespace": "",
}


def load_cfg(cfg_path):
    """
    loads and verifies a config (cfg) based on the path

    cfg_path: str config path
    """
    with open(cfg_path, "r", encoding="utf-8") as fp:
        cfg = json.load(fp)

    for key in DEFAULT_CFG:
        if key not in cfg or len(cfg[key]) == 0:
            raise IOError(
                f"""{key} is empty.
                Use `maestro_cli configure --{key} value` to finalize your enviroment
            """
            )

    return cfg
