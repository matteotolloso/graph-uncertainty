#!/usr/bin/env bash
# Build the image from the repository root (the Dockerfile copies pyproject.toml and cgnn/).
cd "$(dirname "$0")/.." && docker build -f docker/Dockerfile \
    --build-arg USERNAME="$(whoami)" --build-arg USER_ID="$(id -u)" -t cgnn .
