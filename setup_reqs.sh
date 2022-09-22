#!/bin/bash

virtualenv env
source env/bin/activate
pip install -r requirements.txt
pip install --upgrade pip
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
huggingface-cli login
