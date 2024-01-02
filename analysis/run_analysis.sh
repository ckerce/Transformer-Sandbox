#!/bin/bash

for x in `ls config/analysis/*.py`; do python train.py $x; done
