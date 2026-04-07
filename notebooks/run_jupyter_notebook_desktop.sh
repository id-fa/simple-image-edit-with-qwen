#!/bin/bash

PROJECT_NAME="simple-image-edit-qwen"
mkdir $PROJECT_NAME
cd $PROJECT_NAME

# uv環境作成
uv venv --python 3.14

# 環境アクティベート
source .venv/bin/activate

# パッケージインストール
uv pip install \
    jupyter \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    plotly \
    ipywidgets

# Jupyter設定
jupyter notebook --generate-config

# Jupyter起動
echo "Jupyter Notebookを起動します"
jupyter notebook
