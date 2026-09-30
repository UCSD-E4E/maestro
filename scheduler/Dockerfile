FROM quay.io/jupyter/pytorch-notebook:cuda12-python-3.11.8

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /usr/local/bin/uv

# The base image runs as jovyan, which cannot create /app/.venv.
USER root

WORKDIR /app
# Use the image's Python 3.11 instead of downloading one.
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project
ADD . /app

EXPOSE 3000

CMD ["uv", "run", "--locked", "python", "scheduler/label_studio_scheduler.py"]
