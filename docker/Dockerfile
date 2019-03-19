FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# Install necessary packages
RUN apt-get update
RUN apt-get install -y git-core wget bzip2 cmake libgoogle-glog-dev libboost-timer-dev libpng-dev

# Install python interpreter based on miniconda
RUN wget -nv https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O /tmp/miniconda.sh
RUN bash /tmp/miniconda.sh -b -f -p /
ENV PYTHONHOME=/
RUN conda install -y opencv

# Install Python dependencies
RUN pip install tqdm pillow jupyterlab
RUN pip install cupy-cuda100==5.3.0 chainer==5.3.0

# Download ResNet model files, as Chainer doesn't do so by itself
RUN mkdir -p /root/.chainer/dataset/pfnet/chainer/models/
RUN wget -nv "https://onedrive.live.com/download.aspx?cid=4006CBB8476FF777&authKey=%21AAFW2%2DFVoxeVRck&resid=4006CBB8476FF777%2117895&ithint=%2Ecaffemodel" -O /root/.chainer/dataset/pfnet/chainer/models/ResNet-50-model.caffemodel
RUN wget -nv "https://onedrive.live.com/download.aspx?cid=4006CBB8476FF777&authKey=%21AAFW2%2DFVoxeVRck&resid=4006CBB8476FF777%2117896&ithint=%2Ecaffemodel" -O /root/.chainer/dataset/pfnet/chainer/models/ResNet-101-model.caffemodel
RUN wget -nv "https://onedrive.live.com/download.aspx?cid=4006CBB8476FF777&authKey=%21AAFW2%2DFVoxeVRck&resid=4006CBB8476FF777%2117897&ithint=%2Ecaffemodel" -O /root/.chainer/dataset/pfnet/chainer/models/ResNet-152-model.caffemodel

# Let chainer download a VGG model file (this takes a while due to narrow bandwidth)
RUN python -c 'import chainer; chainer.links.VGG16Layers()'


# Install TensorRT
ADD nv-tensorrt-repo-ubuntu1804-cuda10.0-trt5.0.2.6-ga-20181009_1-1_amd64.deb /
RUN dpkg -i nv-tensorrt-repo-ubuntu1804-cuda10.0-trt5.0.2.6-ga-20181009_1-1_amd64.deb
RUN apt-get update
RUN apt-get install -y tensorrt

# Configurations necessay for reproducing chainer-tensorrt
ENV CPATH=${CPATH}:/include:/usr/local/cuda/include
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
ENV LIBRARY_PATH=${LIBRARY_PATH}:/usr/local/cuda/lib64
ENV CPATH=${CPATH}:/include/python3.6m

# Install pybind11, necessary for chainer-trt python interface
WORKDIR /tmp
RUN wget -nv https://github.com/pybind/pybind11/archive/v2.2.1.tar.gz
RUN tar xf v2.2.1.tar.gz
RUN mv pybind11-2.2.1/include/pybind11 /usr/include

# Install google-test, necessary for chainer-trt tests
WORKDIR /tmp
RUN wget https://github.com/google/googletest/archive/release-1.8.0.tar.gz
RUN tar xf release-1.8.0.tar.gz
WORKDIR /tmp/googletest-release-1.8.0
RUN cmake .
RUN make -s
RUN make install

# Install google-benchmark, necessary for chainer-trt micro-benchmarks
WORKDIR /tmp
RUN wget https://github.com/google/benchmark/archive/v1.4.0.tar.gz
RUN tar xf v1.4.0.tar.gz
WORKDIR /tmp/benchmark-1.4.0
RUN cmake -DCMAKE_BUILD_TYPE=RELEASE .
RUN make -s
RUN make install

# Build and install chainer-trt
RUN git clone https://github.com/pfnet-research/chainer-trt.git /root/chainer-trt
WORKDIR /root/chainer-trt
RUN pip install -e .
RUN python test/make_test_cases.py
RUN mkdir -p build
WORKDIR /root/chainer-trt/build
RUN cmake -DWITH_BENCHMARK=on ..
RUN make -s
RUN make install
ENV PYTHONPATH=$PYTHONPATH:/root/chainer-trt:/usr/local/lib

WORKDIR /root/chainer-trt

# Download a test image
RUN wget -nv "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/359px-Cat_November_2010-1a.jpg" -O cat.jpg

ENTRYPOINT "/bin/bash"
