#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" && pwd )"
SRC_DIR="$( dirname ${DIR} )"
PROTOS=${DIR}/*.proto

python3 -m grpc_tools.protoc -I${DIR} ${PROTOS} --python_out=${SRC_DIR} --grpc_python_out=${SRC_DIR}
