import os
import sys

from invoke import Context, task

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

WINDOWS = os.name == "nt"
PROJECT_NAME = "alcohol_classifier"
PYTHON_VERSION = "3.12"


# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(
        f"python src/{PROJECT_NAME}/data.py dataset.path_raw=data/raw dataset.path_processed=data/processed",
        echo=True,
        pty=not WINDOWS
    )

@task
def train(c, lr=None, epochs=None, batch=None, freeze=False, not_pretrained=False):
    """
    Run training with shorter arguments.
    Example: inv train --lr 0.005 --epochs 20 --freeze
    """
    cmd = "python -m src.alcohol_classifier.train"

    # Map short flags to long Hydra overrides
    if lr:
        cmd += f" model.lr={lr}"
    if epochs:
        cmd += f" model.epochs={epochs}"
    if batch:
        cmd += f" dataset.batch_size={batch}"
    if freeze:
        cmd += " model.freeze_backbone=True"
    if not_pretrained:
        cmd += " model.pretrained=False"

    print(f"🚀 Running: {cmd}")
    c.run(cmd, out_stream=sys.stdout, pty=not WINDOWS, encoding="utf-8")


@task
def evaluate(c):
    """Run evaluation on the best saved model."""
    os.environ["PYTHONUTF8"] = "1"
    cmd = "python -m src.alcohol_classifier.evaluate"

    print(f"🚀 Running: {cmd}")
    c.run(cmd, out_stream=sys.stdout, pty=not WINDOWS, encoding="utf-8")


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )

# API commands
@task
def api(ctx: Context, reload: bool = True) -> None:
    """Start the FastAPI backend server."""
    reload_str = "--reload" if reload else ""
    # We use api.api:app because your file is api/api.py
    # and the FastAPI object is named 'app'
    print("🚀 Starting FastAPI backend on http://127.0.0.1:8000")
    ctx.run(f"python -m uvicorn api.api:app {reload_str} --port 8000", echo=True, pty=not WINDOWS)

@task
def frontend(ctx: Context, backend_url: str = "http://127.0.0.1:8000") -> None:
    """Start the Streamlit frontend."""
    # This sets the BACKEND env var for the duration of this command
    os.environ["BACKEND"] = backend_url
    print(f"🖥️  Starting Streamlit frontend pointing to {backend_url}")
    ctx.run("python -m streamlit run api/frontend.py", echo=True, pty=not WINDOWS)

# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
