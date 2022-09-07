#!/bin/bash --login


set +euo pipefail

# setup python env
conda create -y --name myenv python=3.7
conda activate myenv
conda install -y -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

python3 -m pip install --upgrade pip wheel
pip3 install -r requirements.txt

# download pretrained
mkdir -p /app/models
gdown https://drive.google.com/file/d/1-slos4_7v9bMOYFEs40HKJFJ4GI8BfzJ/view?usp=sharing --fuzzy -O /app/models/yolo.pt
gdown https://drive.google.com/file/d/1-U253UBmypqAZDRJZ2fgC3hQE0-ZBSAJ/view?usp=sharing --fuzzy -O /app/models/resnet50.h5
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# enable strict mode:
set -euo pipefail
