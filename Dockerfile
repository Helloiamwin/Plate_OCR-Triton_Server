FROM nvcr.io/nvidia/tritonserver:22.05-py3
RUN apt update
RUN pip install opencv-python
RUN pip install scikit-image
RUN pip install pyclipper shapely
RUN apt-get install libgl1 -y