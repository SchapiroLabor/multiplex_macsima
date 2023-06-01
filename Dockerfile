FROM python:3.10.11-slim-buster
COPY requirements.txt requirements.txt
#COPY imagecodecs-2020.5.30.tar.gz imagecodecs-2020.5.30.tar.gz
#RUN apt update
#RUN apt install -y build-essential 
#RUN apt-get install -y manpages-dev
#RUN echo gcc --version
RUN python3 -m pip install --upgrade pip 
#RUN pip install --upgrade wheel
RUN python3 -m pip install --upgrade setuptools
#RUN apt install -y libgif7 libtiff5 libsnappy1v5 libwebp6 libjbig0 libblosc1 libzopfli1 libopenjp2-7 liblcms2-2 libaec0 libbrotli1 libjxr0
#RUN python3 -m pip install Cython
#RUN python3 -m pip install imagecodecs==2023.3.16
#RUN apt update
#RUN apt install -y build-essential 
#RUN apt-get install -y manpages-dev
#RUN echo gcc --version
#RUN python3 -m pip install --upgrade pip setuptools
#RUN python3 -m pip install --no-cache-dir --upgrade imagecodecs
#RUN pip install --no-cache-dir --upgrade -r requirements.txt
#RUN apt-get update && \
#    apt-get install -y gcc
RUN pip install --no-cache-dir --upgrade -r requirements.txt
#ENV PATH="${PATH}:/opt/conda/bin"
WORKDIR /stack_macsima
COPY . .


