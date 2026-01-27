# Base image
FROM nvcr.io/nvidia/pyg:24.05-py3

# Set build arguments for user and user ID
ARG USERNAME
ARG USER_ID

# Install sudo and create user dynamically
RUN apt-get update && apt-get install -y sudo && \
    useradd -m -s /bin/bash -u $USER_ID $USERNAME && \
    echo "$USERNAME:pass" | chpasswd && \
    usermod -aG sudo $USERNAME

RUN echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chown root:root /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME


# Set the working directory inside the container
WORKDIR /workspace

# Copy only the requirements file
COPY requirements-docker.txt /workspace/

# Ensure the new user owns the workspace
RUN chown -R $USERNAME:$USERNAME /workspace

# Switch to the non-root user
USER $USERNAME

# Add ~/.local/bin to PATH to avoid pip warnings
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

# Install Python dependencies as the new user
RUN python -m pip install --upgrade pip && \
    pip install  --no-cache-dir --user -r ./requirements-docker.txt

RUN python -m pip install torch_sparse torch_scatter
# Set default command

CMD ["/bin/bash"]
