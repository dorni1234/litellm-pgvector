FROM mcr.microsoft.com/devcontainers/python:3-3.12-trixie

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    postgresql-client \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*