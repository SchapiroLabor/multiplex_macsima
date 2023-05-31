FROM python:3.10.11-slim-buster
COPY requirements.txt requirements.txt
RUN apt-get update && \
    apt-get install gcc
RUN pip install --no-cache-dir --upgrade -r requirements.txt
#ENV PATH="${PATH}:/opt/conda/bin"
WORKDIR /stack_macsima
COPY . .


