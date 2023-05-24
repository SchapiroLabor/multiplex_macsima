FROM mambaorg/micromamba:latest
COPY --chown=$MAMBA_USER:$MAMBA_USER conda_enviroment.yml /tmp/conda_enviroment.yaml
RUN micromamba install -y -n base -f /tmp/conda_enviroment.yaml && \
    micromamba clean --all --yes
ENV PATH="${PATH}:/opt/conda/bin"
WORKDIR /stack_macsima
COPY . .