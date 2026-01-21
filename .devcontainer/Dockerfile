FROM mcr.microsoft.com/devcontainers/python:3.9

RUN apt-get update && apt-get install -y \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy \
    scipy \
    pillow \
    matplotlib \
    jupyter \
    imageio \
    future
