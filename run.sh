#!/bin/bash

rm -rf SaveKPImg
rm -rf SaveReconstructedImg
rm -rf SavetfKPImg
#python3 train_noTFloss.py
#python3 train_R2D2.py
#python3 train.py
#python3 train_2.py
#python3 train_3.py
#python3 train_5.py
#python3 train_DETR.py
#python3 train_4.py
python3 train/train_DETR3.py
