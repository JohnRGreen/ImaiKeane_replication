#!/usr/bin/env bash

# set working directory
cd "$(dirname "$0")"

# install dependencies
pip install -r ./binder/requirements.txt

# then call the script
python ./do_all.py
python --version