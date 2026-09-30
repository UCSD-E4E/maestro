# Install Instructions

1) Install python=^3.10
2) Install pipx. See [https://github.com/pypa/pipx](https://github.com/pypa/pipx)
3) make sure to run `pipx ensurepath`
4) run `python -m pipx install "git+ssh://git@github.com/UCSD-E4E/maestro.git#subdirectory=cli"` (note you MUST have ssh keys for private UCSD-E4E repos)
5) test with running `maestro_cli`   

# Development Instructions
1) Install [uv](https://docs.astral.sh/uv/)
2) From the repo root, run `uv sync --package maestro-cli`
3) Live development: `uv run maestro_cli CMDs/ARGS`
