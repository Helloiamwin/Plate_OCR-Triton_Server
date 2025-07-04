docker run --gpus=all  --shm-size=2G --rm \
 -p8000:8000 -p8001:8001 -p8002:8002 \
 -v $PWD/triton_models_new:/models \
 22.05-py3_new tritonserver --model-repository=/models --strict-model-config=false
 

#--log-verbose 2 --strict-model-config=false

#nvcr.io/nvidia/tritonserver:22.05-py3_new tritonserver --model-repository=/models \
#  --log-verbose 1
