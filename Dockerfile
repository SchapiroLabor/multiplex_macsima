FROM mambaorg/micromamba:latest
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
ENV PATH="${PATH}:/opt/conda/bin"
WORKDIR /staging
COPY . .