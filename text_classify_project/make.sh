# /bin/sh

pip install -r algorithm_grpc/abcft_algorithm_grpc/requirements.txt
pip install -r forecast_extraction/abcft_algorithm_forecast_extraction/requirements.txt


export PYTHONPATH=$PYTHONPATH:$PWD/algorithm_grpc
export PYTHONPATH=$PYTHONPATH:$PWD/forecast_extraction

