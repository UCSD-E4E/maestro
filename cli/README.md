# Install Instructions

1) Install python=^3.10
2) Install pipx. See [https://github.com/pypa/pipx](https://github.com/pypa/pipx)
3) make sure to run `pipx ensurepath`
4) run `python -m pipx install git+https://github.com/UCSD-E4E/maestro_cli.git` (note you MUST have ssh keys for private UCSD-E4E repos)
5) test with running `maestro_cli`   


# Development Instructions
1) Install python=^3.10
2) Install Poetry. and cd into the maestro_cli folder
3) run poetry install
4) Live development:  `poetry run maestro_cli CMDs/ARGS`
