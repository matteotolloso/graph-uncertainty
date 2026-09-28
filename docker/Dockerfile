# Base image with PyTorch + PyG (CUDA). Build from the repository root:
#   docker build -f docker/Dockerfile --build-arg USERNAME=$(whoami) --build-arg USER_ID=$(id -u) -t cgnn .
FROM nvcr.io/nvidia/pyg:24.05-py3

ARG USERNAME
ARG USER_ID

RUN apt-get update && apt-get install -y sudo && \
    useradd -m -s /bin/bash -u $USER_ID $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

WORKDIR /workspace
RUN chown -R $USERNAME:$USERNAME /workspace
USER $USERNAME
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

# Install the package metadata/deps only; the source is mounted at runtime (docker-run.sh).
COPY --chown=$USERNAME:$USERNAME pyproject.toml README.md /tmp/cgnn/
COPY --chown=$USERNAME:$USERNAME cgnn /tmp/cgnn/cgnn
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --user "/tmp/cgnn[knn,ogb,esn,results,dev]" && \
    pip install --no-cache-dir --user torch_sparse torch_scatter && \
    pip uninstall -y cgnn

CMD ["/bin/bash"]
