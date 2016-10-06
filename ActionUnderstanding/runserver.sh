#!/bin/bash
export FLASK_APP=ActionUnderstanding/RNN-SRD.py
python -m flask run --port=7272 --host=0.0.0.0
