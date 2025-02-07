"""
Handler for CLI
"""

import os
import json
import uuid
import click

from maestro_cli.util import DEFAULT_CFG, load_cfg
from maestro_cli.control_pannel import (
    spin_up_jobs,
    spin_down_jobs,
    list_all_kube_objects,
)

## This should be in some cache. How to set cache?
## Location of config file for user testing
CFG_PATH = "config.json"


## Holds the list of subcommands
@click.group()
def cli():
    """
    Placeholder for holding a group of commands
    See every @cli.comand()
    """


@cli.command()
@click.argument("spin_direction")
@click.option(
    "--storage",
    is_flag=True,
    show_default=True,
    default=True,
    help="If PVC should be cleaned",
)
def spin(spin_direction, storage):
    """Handles the start stop for kubernetes

    SPIN_DIRECTION is up or down
    """
    click.echo(f"spining {spin_direction}")
    cfg = load_cfg(CFG_PATH)
    if spin_direction == "up":
        spin_up_jobs(cfg)
    else:
        spin_down_jobs(cfg, keep_storage=storage)
        if not storage:
            click.echo("warning: persistant storage can take a while to remove")


@cli.command()
def ls():
    """
    Prints what kubernetes objects are running
    """
    cfg = load_cfg(CFG_PATH)
    list_all_kube_objects(cfg)


@cli.command()
def env():
    """Displays configuration. To set configuration see
    `maestro_cli configure --help`
    """
    # https://stackoverflow.com/questions/12943819/how-to-prettyprint-a-json-file
    click.echo(json.dumps(load_cfg(CFG_PATH), indent=4))


@cli.command()
@click.option("--cfg_file", default="", show_default=False)
@click.option("--scheduler_path", default="", show_default=False)
@click.option("--trainer_path", default="", show_default=False)
@click.option("--ingress_url", default="", show_default=False)
@click.option("--namespace", default="", show_default=False)
@click.option("--name", default="", show_default=False)
def configure(cfg_file, scheduler_path, trainer_path, ingress_url, namespace, name):
    """
    Sets up env for dev/prod machine. These variables allow for creating seperate working
    enviroments for developers
    """
    click.echo(os.getcwd())
    # os.remove(CFG_PATH)
    if os.path.exists(CFG_PATH):
        with open(CFG_PATH, "r", encoding="utf-8") as fp:
            cfg = json.load(fp)
            if "UUID" not in cfg:
                cfg["UUID"] = str(uuid.uuid4())
    else:
        cfg = DEFAULT_CFG

    cfg_new = {}

    if len(cfg_file) != 0:
        with open(cfg_file, "r", encoding="utf-8") as fp:
            cfg_new = json.load(fp)

    # Update values
    if len(scheduler_path) != 0:
        cfg_new["scheduler_path"] = scheduler_path
    if len(trainer_path) != 0:
        cfg_new["trainer_path"] = trainer_path
    if len(ingress_url) != 0:
        cfg_new["ingress_url"] = ingress_url
    if len(namespace) != 0:
        cfg_new["namespace"] = namespace
    if len(name) != 0:
        cfg_new["UUID"] = name

    for key in cfg_new.keys():
        cfg[key] = cfg_new[key]

    click.echo("new file")
    with open(CFG_PATH, "w", encoding="utf-8") as fp:
        json.dump(cfg, fp)

    load_cfg(CFG_PATH)


if __name__ == "__main__":
    cli()
