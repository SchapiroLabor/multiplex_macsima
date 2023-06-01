FROM python:3.10.11-slim-bullseye
COPY requirements.txt requirements.txt
RUN apt update
RUN apt install -y build-essential 
RUN apt-get install -y manpages-dev
RUN echo gcc --version
RUN python3 -m pip install --upgrade pip setuptools
RUN python3 -m pip install --no-cache-dir --upgrade imagecodecs
#RUN pip install --no-cache-dir --upgrade -r requirements.txt
#RUN apt-get update && \
#    apt-get install -y gcc
#RUN pip install --no-cache-dir --upgrade -r requirements.txt
#ENV PATH="${PATH}:/opt/conda/bin"
WORKDIR /stack_macsima
COPY . .


