FROM python:3.10.11-slim-buster
COPY requirements.txt requirements.txt
COPY imagecodecs-2023.3.16-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl imagecodecs-2023.3.16-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN python3 -m pip install imagecodecs-2023.3.16-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
#RUN apt update
#RUN apt install -y build-essential 
#RUN apt-get install -y manpages-dev
#RUN echo gcc --version
#RUN python3 -m pip install --upgrade pip setuptools
#RUN python3 -m pip install --no-cache-dir --upgrade imagecodecs
#RUN pip install --no-cache-dir --upgrade -r requirements.txt
#RUN apt-get update && \
#    apt-get install -y gcc
#RUN pip install --no-cache-dir --upgrade -r requirements.txt
#ENV PATH="${PATH}:/opt/conda/bin"
WORKDIR /stack_macsima
COPY . .


